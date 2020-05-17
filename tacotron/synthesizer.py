import os
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd
import tensorflow as tf
from tacotron.datasets import audio
from tacotron.utils.infolog import log
from librosa import effects
from tacotron.models import create_model
from tacotron.utils import plot
from tacotron.utils.text import text_to_sequence
import os

class Synthesizer:
    def load(self, checkpoint_path, hparams, gta=False, model_name='Tacotron'):
        log('Constructing model: %s' % model_name)
        #Force the batch size to be known in order to use attention masking in batch synthesis
        inputs = tf.placeholder(tf.int32, (1, None), name='inputs')
        input_lengths = tf.placeholder(tf.int32, (1), name='input_lengths')

        targets = tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_targets')
        split_infos = tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name='split_infos')
        with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
            self.model = create_model(model_name, hparams)
            if gta:
                self.model.initialize(inputs, input_lengths, targets, gta=gta)
            else:
                self.model.initialize(inputs, input_lengths)

            self.mel_outputs = self.model.mel_outputs
            if hparams.predict_linear:
                self.linear_outputs = self.model.linear_outputs 
                
            self.alignments = self.model.alignments
            self.stop_token_prediction = self.model.stop_token_prediction
            self.targets = targets

        self.gta = gta
        self._hparams = hparams
        #pad input sequences with the <pad_token> 0 ( _ )
        self._pad = 0
        #explicitely setting the padding to a value that doesn't originally exist in the spectogram
        #to avoid any possible conflicts, without affecting the output range of the model too much
        if hparams.symmetric_mels:
            self._target_pad = -hparams.max_abs_value
        else:
            self._target_pad = 0.

        self.inputs = inputs
        self.input_lengths = input_lengths
        self.targets = targets
        self.split_infos = split_infos

        log('Loading checkpoint: %s' % checkpoint_path)
        #Memory allocation on the GPUs as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)


    def synthesize(self, text, step, out_dir, log_dir, mel_filenames, cwd):
        hparams = self._hparams
        #[-max, max] or [0,max]
        T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (0, hparams.max_abs_value)

        inputs = [np.asarray(text_to_sequence(text))]
        input_lengths = [len(inputs[0])]

        feed_dict = {
            self.inputs: np.asarray(inputs, dtype=np.int32),
            self.input_lengths: np.asarray(input_lengths, dtype=np.int32),
        }
        
        if hparams.predict_linear:
            linears, mels, alignments, stop_tokens = self.session.run([self.linear_outputs, self.mel_outputs, self.alignments, self.stop_token_prediction], feed_dict=feed_dict)
            linear = linears[0]
        else:
            mels, alignments, stop_tokens = self.session.run([self.mel_outputs, self.alignments, self.stop_token_prediction], feed_dict=feed_dict)
        
        new_out = os.path.join(cwd, 'logs-Tacotron-2/wav_plot')

        #Linearize outputs (1D arrays)
        mel = mels[0]
        alignment = alignments[0]
        stop_token = np.round(stop_tokens[0]).tolist()
        target_length = stop_token.index(1) if 1 in stop_token else len(stop_token)

        mel = mel[:target_length, :]
        if hparams.predict_linear:
            linear = linear[:target_length, :]
            linear = np.clip(linear, T2_output_range[0], T2_output_range[1])
            
            wav = audio.inv_linear_spectrogram(linear.T, hparams)
            audio.save_wav(wav, os.path.join(new_out, 'eval-step_{}-from_linear.wav'.format(step)), sr=hparams.sample_rate)
            
        mel = np.clip(mel, T2_output_range[0], T2_output_range[1])
        if not hparams.predict_linear:
            wav = audio.inv_mel_spectrogram(mel.T, hparams)
            audio.save_wav(wav, os.path.join(new_out, 'eval-step_{}-from_mel.wav'.format(step)), sr=hparams.sample_rate)

        text = text[:30]

        plot.plot_alignment(alignment, os.path.join(new_out, 'eval-step_{}-alignment.png'.format(step)),
            title='{}'.format(text), split_title=True, max_len=target_length)

        #save mel spectrogram plot
        plot.plot_spectrogram(mel, os.path.join(new_out, 'eval-step_{}-mel.png'.format(step)),
            title='{}'.format(text), split_title=True)
        
        print('step: {}'.format(step))
        print('pyin: {}'.format(text))

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    def _prepare_inputs(self, inputs):
        max_len = max([len(x) for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

    def _prepare_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_target(t, data_len) for t in targets]), data_len

    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

    def _get_output_lengths(self, stop_tokens):
        #Determine each mel length by the stop token predictions. (len = first occurence of 1 in stop_tokens row wise)
        output_lengths = [row.index(1) if 1 in row else len(row) for row in np.round(stop_tokens).tolist()]
        return output_lengths
