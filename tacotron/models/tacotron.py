import tensorflow as tf 
from tacotron.utils.symbols import symbols
from tacotron.utils.infolog import log
from tacotron.models.helpers import TacoTrainingHelper, TacoTestHelper
from tacotron.models.modules import *
from tensorflow.contrib.seq2seq import dynamic_decode
from tacotron.models.Architecture_wrappers import TacotronEncoderCell, TacotronDecoderCell
from tacotron.models.custom_decoder import CustomDecoder
from tacotron.models.attention import ForwardLocationSensitiveAttention

import numpy as np

def split_func(x, split_pos):
    rst = []
    start = 0
    # x will be a numpy array with the contents of the placeholder below
    for i in range(split_pos.shape[0]):
        rst.append(x[:,start:start+split_pos[i]])
        start += split_pos[i]
    return rst

class Tacotron():
    """Tacotron-2 Feature prediction Model.
    """
    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, inputs, input_lengths, mel_targets=None, stop_token_targets=None, linear_targets=None, targets_lengths=None, gta=False,
            global_step=None, is_training=False, is_evaluating=False):
        
        hp = self._hparams 
        batch_size = tf.shape(inputs)[0]
        gta = False 
        self.num_atten = 5

        T2_output_range = (-hp.max_abs_value, hp.max_abs_value) if hp.symmetric_mels else (0, hp.max_abs_value)

        with tf.variable_scope('inference') as scope:
            assert hp.tacotron_teacher_forcing_mode in ('constant', 'scheduled')
            if hp.tacotron_teacher_forcing_mode == 'scheduled' and is_training:
                assert global_step is not None

            # Embeddings ==> [batch_size, sequence_length, embedding_dim]
            self.embedding_table = tf.get_variable(
                'inputs_embedding', [len(symbols), hp.embedding_dim], dtype=tf.float32)

            embedded_inputs = tf.nn.embedding_lookup(self.embedding_table, inputs)

            #Encoder Cell ==> [batch_size, encoder_steps, encoder_lstm_units]
            encoder_cell = TacotronEncoderCell(
                EncoderConvolutions(is_training, hparams=hp, scope='encoder_convolutions'),
                EncoderRNN(is_training, size=hp.encoder_lstm_units,
                    zoneout=hp.tacotron_zoneout_rate, scope='encoder_LSTM'))

            self.encoder_outputs = encoder_cell(embedded_inputs, input_lengths)

            #For shape visualization purpose
            self.enc_conv_output_shape = encoder_cell.conv_output_shape

            #Decoder Parts
            #Attention Decoder Prenet
            prenet = Prenet(is_training, layers_sizes=hp.prenet_layers, drop_rate=hp.tacotron_dropout_rate, scope='decoder_prenet')
            #Attention Mechanism

            attention_mechanism = ForwardLocationSensitiveAttention(hp.attention_dim, self.encoder_outputs, 
                                hparams=hp, is_training=is_training or is_evaluating, memory_sequence_length=input_lengths, 
                                smoothing=hp.smoothing)
            
            #Decoder LSTM Cells
            decoder_lstm = DecoderRNN(is_training, layers=hp.decoder_layers,
                size=hp.decoder_lstm_units, zoneout=hp.tacotron_zoneout_rate, scope='decoder_LSTM')
            #Frames Projection layer
            frame_projection = FrameProjection(hp.num_mels * hp.outputs_per_step, scope='linear_transform_projection')
            #<stop_token> projection layer
            stop_projection = StopProjection(is_training or is_evaluating, shape=hp.outputs_per_step, scope='stop_token_projection')


            #Decoder Cell ==> [batch_size, decoder_steps, num_mels * r] (after decoding)
            decoder_cell = TacotronDecoderCell(
                prenet,
                attention_mechanism,
                decoder_lstm,
                frame_projection,
                stop_projection)

            #Define the helper for our decoder
            if is_training or is_evaluating or gta:
                self.helper = TacoTrainingHelper(batch_size, mel_targets, hp, gta, is_evaluating, global_step)
            else:
                self.helper = TacoTestHelper(batch_size, hp, input_lengths)

            #initial decoder state
            decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            #Only use max iterations at synthesis time
            max_iters = hp.max_iters if not (is_training or is_evaluating) else None

            #Decode
            (frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(
                CustomDecoder(decoder_cell, self.helper, decoder_init_state),
                impute_finished=False,
                maximum_iterations=max_iters,
                swap_memory=hp.tacotron_swap_with_cpu)


            # Reshape outputs to be one output per entry 
            #==> [batch_size, non_reduced_decoder_steps (decoder_steps * r), num_mels]
            self.decoder_output = tf.reshape(frames_prediction, [batch_size, -1, hp.num_mels])
            self.stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])

            if hp.clip_outputs:
                self.decoder_output = tf.minimum(tf.maximum(self.decoder_output, T2_output_range[0] - hp.lower_bound_decay), T2_output_range[1])

            #Postnet
            postnet = Postnet(is_training, hparams=hp, scope='postnet_convolutions')

            #Compute residual using post-net ==> [batch_size, decoder_steps * r, postnet_channels]
            residual = postnet(self.decoder_output)

            #Project residual to same dimension as mel spectrogram 
            #==> [batch_size, decoder_steps * r, num_mels]
            residual_projection = FrameProjection(hp.num_mels, scope='postnet_projection')
            self.projected_residual = residual_projection(residual)

            #Compute the mel spectrogram
            self.mel_outputs = self.decoder_output + self.projected_residual

            if hp.clip_outputs:
                self.mel_outputs = tf.minimum(tf.maximum(self.mel_outputs, T2_output_range[0] - hp.lower_bound_decay), T2_output_range[1])

            if hp.predict_linear:
                # Add post-processing CBHG. This does a great job at extracting features from mels before projection to Linear specs.
                post_cbhg = CBHG(hp.cbhg_kernels, hp.cbhg_conv_channels, hp.cbhg_pool_size, [hp.cbhg_projection, hp.num_mels],
                                hp.cbhg_projection_kernel_size, hp.cbhg_highwaynet_layers, hp.cbhg_highway_units, 
                                hp.cbhg_rnn_units, hp.batch_norm_position, is_training, name='CBHG_postnet')

                #[batch_size, decoder_steps(mel_frames), cbhg_channels]
                self.post_outputs = post_cbhg(self.mel_outputs, None)

                #Linear projection of extracted features to make linear spectrogram
                linear_specs_projection = FrameProjection(hp.num_freq, scope='cbhg_linear_specs_projection')

                #[batch_size, decoder_steps(linear_frames), num_freq]
                self.linear_outputs = linear_specs_projection(self.post_outputs)

                if hp.clip_outputs:
                    self.linear_outputs = tf.minimum(tf.maximum(self.linear_outputs, T2_output_range[0] - hp.lower_bound_decay), T2_output_range[1])

            #Grab alignments from the final decoder state
            self.alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

            log('initialisation done.')

        if is_training:
            self.ratio = self.helper._ratio

        self.inputs = inputs 
        self.input_lengths = input_lengths
        self.mel_targets = mel_targets
        self.linear_targets = linear_targets 
        self.targets_lengths = targets_lengths 
        self.stop_token_targets = stop_token_targets
        self.gta = gta 
        self.all_vars = tf.trainable_variables()
        self.is_training = is_training 
        self.is_evaluating = is_evaluating 
        self.fine_tune_params = [v for v in self.all_vars if not ('inputs_embedding' in v.name or 'encoder_' in v.name)]

        self.final_params = self.all_vars if not hp.tacotron_fine_tuning else self.fine_tune_params

        log('Initialized Tacotron model. Dimensions (? = dynamic shape): ')
        log('  Train mode:               {}'.format(is_training))
        log('  Eval mode:                {}'.format(is_evaluating))
        log('  GTA mode:                 {}'.format(gta))
        log('  Synthesis mode:           {}'.format(not (is_training or is_evaluating)))
        log('  Input:                    {}'.format(inputs.shape))
        log('  embedding:                {}'.format(embedded_inputs.shape))
        log('  enc conv out:             {}'.format(self.enc_conv_output_shape))
        log('  encoder out:              {}'.format(self.encoder_outputs.shape))
        log('  decoder out:              {}'.format(self.decoder_output.shape))
        log('  residual out:             {}'.format(residual.shape))
        log('  projected residual out:   {}'.format(self.projected_residual.shape))
        log('  mel out:                  {}'.format(self.mel_outputs.shape))
        if hp.predict_linear:
            log('  linear out:               {}'.format(self.linear_outputs.shape))
        
        log('  <stop_token> out:         {}'.format(self.stop_token_prediction.shape))

        #1_000_000 is causing syntax problems for some people?! Python please :)
        log('  Tacotron Parameters       {:.3f} Million.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1000000))
        log(' fine tune paarmaters:      {:.3f} Million.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in self.fine_tune_params]) / 1000000))
        log(' final  paarmaters:      {:.3f} Million.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in self.final_params]) / 1000000))
    

    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        hp = self._hparams

        with tf.variable_scope('loss') as scope:
            if hp.mask_decoder:
                # Compute loss of predictions before postnet
                self.before_loss = MaskedMSE(self.mel_targets, self.decoder_output, self.targets_lengths,
                    hparams=self._hparams)
                # Compute loss after postnet
                self.after_loss = MaskedMSE(self.mel_targets, self.mel_outputs, self.targets_lengths,
                    hparams=self._hparams)
                #Compute <stop_token> loss (for learning dynamic generation stop)
                self.stop_token_loss = MaskedSigmoidCrossEntropy(self.stop_token_targets,
                    self.stop_token_prediction, self.targets_lengths, hparams=self._hparams)
                #Compute masked linear loss

                if hp.predict_linear:
                    #Compute Linear L1 mask loss (priority to low frequencies)
                    self.linear_loss = MaskedLinearLoss(self.linear_targets, self.linear_outputs,
                            self.targets_lengths, hparams=self._hparams)

            else:
                # Compute loss of predictions before postnet
                self.before_loss = tf.losses.mean_squared_error(self.mel_targets, self.decoder_output)
                # Compute loss after postnet
                self.after_loss = tf.losses.mean_squared_error(self.mel_targets, self.mel_outputs)
                #Compute <stop_token> loss (for learning dynamic generation stop)
                self.stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.stop_token_targets,
                    logits=self.stop_token_prediction))

                if hp.predict_linear:
                    #Compute linear loss
                    #From https://github.com/keithito/tacotron/blob/tacotron2-work-in-progress/models/tacotron.py
                    #Prioritize loss for frequencies under 2000 Hz.
                    l1 = tf.abs(self.linear_targets - self.linear_outputs)
                    n_priority_freq = int(2000 / (hp.sample_rate * 0.5) * hp.num_freq)
                    self.linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:,:,0:n_priority_freq])


            # Compute the regularization weight
            if hp.tacotron_scale_regularization:
                reg_weight_scaler = 1. / (2 * hp.max_abs_value) if hp.symmetric_mels else 1. / (hp.max_abs_value)
                reg_weight = hp.tacotron_reg_weight * reg_weight_scaler
            else:
                reg_weight = hp.tacotron_reg_weight

            # Regularize variables
            # Exclude all types of bias, RNN (Bengio et al. On the difficulty of training recurrent neural networks), embeddings and prediction projection layers.
            # Note that we consider attention mechanism v_a weights as a prediction projection layer and we don't regularize it. (This gave better stability)
            self.regularization_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_vars
                if not('bias' in v.name or 'Bias' in v.name or '_projection' in v.name or 'inputs_embedding' in v.name
                    or 'RNN' in v.name or 'LSTM' in v.name)]) * reg_weight

            self.loss = self.before_loss + self.after_loss + self.stop_token_loss + self.regularization_loss 

            if hp.predict_linear:
                self.loss += self.linear_loss
            
    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.
        Args:
            global_step: int32 scalar Tensor representing current global step in training
        '''
        hp = self._hparams

        with tf.variable_scope('optimizer') as scope:
            if hp.tacotron_decay_learning_rate:
                self.decay_steps = hp.tacotron_decay_steps
                self.decay_rate = hp.tacotron_decay_rate
                self.learning_rate = self._learning_rate_decay(hp.tacotron_initial_learning_rate, global_step)
            else:
                self.learning_rate = tf.convert_to_tensor(hp.tacotron_initial_learning_rate)

            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.tacotron_adam_beta1,
                hp.tacotron_adam_beta2, hp.tacotron_adam_epsilon)

            gradients, variables = zip(*optimizer.compute_gradients(self.loss, var_list=self.final_params))
            self.gradients = gradients

            #https://github.com/Rayhane-mamah/Tacotron-2/issues/11
            if hp.tacotron_clip_gradients:
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.) # __mark 0.5 refer
            else:
                clipped_gradients = gradients

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                    global_step=global_step)


    def _learning_rate_decay(self, init_lr, global_step):
        #################################################################
        # Narrow Exponential Decay:

        # Phase 1: lr = 1e-3
        # We only start learning rate decay after 50k steps

        # Phase 2: lr in ]1e-5, 1e-3[
        # decay reach minimal value at step 310k

        # Phase 3: lr = 1e-5
        # clip by minimal learning rate value (step > 310k)
        #################################################################
        hp = self._hparams

        #Compute natural exponential decay
        lr = tf.train.exponential_decay(init_lr, 
            global_step - hp.tacotron_start_decay, #lr = 1e-3 at step 50k
            self.decay_steps, 
            self.decay_rate, #lr = 1e-5 around step 310k
            name='lr_exponential_decay')


        #clip learning rate by max and min values (initial and final values)
        return tf.minimum(tf.maximum(lr, hp.tacotron_final_learning_rate), init_lr)
        
