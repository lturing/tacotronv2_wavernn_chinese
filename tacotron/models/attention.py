#https://github.com/mozilla/TTS/blob/master/layers/common_layers.py

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops, math_ops, nn_ops, variable_scope


def _location_sensitive_score(W_query, W_fil, W_keys):
    """Impelements Bahdanau-style (cumulative) scoring function.
    This attention is described in:
        J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
      gio, “Attention-based models for speech recognition,” in Ad-
      vances in Neural Information Processing Systems, 2015, pp.
      577–585.

    #############################################################################
              hybrid attention (content-based + location-based)
                               f = F * α_{i-1}
       energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f) + b_a))
    #############################################################################

    Args:
        W_query: Tensor, shape '[batch_size, 1, attention_dim]' to compare to location features.
        W_location: processed previous alignments into location features, shape '[batch_size, max_time, attention_dim]'
        W_keys: Tensor, shape '[batch_size, max_time, attention_dim]', typically the encoder outputs.
    Returns:
        A '[batch_size, max_time]' attention score (energy)
    """
    # Get the number of hidden units from the trailing dimension of keys
    dtype = W_query.dtype
    num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

    v_a = tf.get_variable(
        'attention_variable_projection', shape=[num_units], dtype=dtype,
        initializer=tf.contrib.layers.xavier_initializer())
    b_a = tf.get_variable(
        'attention_bias', shape=[num_units], dtype=dtype,
        initializer=tf.zeros_initializer())

    return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])

def _smoothing_normalization(e):
    """Applies a smoothing normalization function instead of softmax
    Introduced in:
        J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
      gio, “Attention-based models for speech recognition,” in Ad-
      vances in Neural Information Processing Systems, 2015, pp.
      577–585.

    ############################################################################
                        Smoothing normalization function
                a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
    ############################################################################

    Args:
        e: matrix [batch_size, max_time(memory_time)]: expected to be energy (score)
            values of an attention mechanism
    Returns:
        matrix [batch_size, max_time]: [0, 1] normalized alignments with possible
            attendance to multiple memory time steps.
    """
    return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)


class ForwardLocationSensitiveAttention(BahdanauAttention):
    """Impelements Bahdanau-style (cumulative) scoring function.
    Usually referred to as "hybrid" attention (content-based + location-based)
    Extends the additive attention described in:
    "D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine transla-
  tion by jointly learning to align and translate,” in Proceedings
  of ICLR, 2015."
    to use previous alignments as additional location features.

    This attention is described in:
    J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
  gio, “Attention-based models for speech recognition,” in Ad-
  vances in Neural Information Processing Systems, 2015, pp.
  577–585.
    """

    def __init__(self,
                 num_units,
                 memory,
                 hparams,
                 is_training,
                 memory_sequence_length=None,
                 smoothing=False,
                 name='ForwardLocationSensitiveAttention'):
        #Create normalization function
        #Setting it to None defaults in using softmax
        normalization_function = _smoothing_normalization if (smoothing == True) else None
        super(ForwardLocationSensitiveAttention, self).__init__(
                num_units=num_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                probability_fn=normalization_function,
                name=name)

        self.location_convolution = tf.layers.Conv1D(filters=hparams.attention_filters,
            kernel_size=hparams.attention_kernel, padding='same', use_bias=True,
            bias_initializer=tf.zeros_initializer(), name='location_features_convolution')
        
        self.location_layer = tf.layers.Dense(units=num_units, use_bias=False,
            dtype=tf.float32, name='location_features_layer')
        
        self.synthesis_constraint = hparams.synthesis_constraint and not is_training
        self.attention_win_size = tf.convert_to_tensor(hparams.attention_win_size, dtype=tf.int32)
        self.constraint_type = hparams.synthesis_constraint_type
        self.memory_sequence_length = memory_sequence_length
        self.is_training = is_training 
        self.init_alpha = tf.concat([tf.reshape(tf.ones_like(memory[:, 0, 0]), (-1, 1)), 
                                        tf.zeros_like(memory[:, :, 0])[:, 1:]], axis=-1)

        self.init_mu = tf.reshape(tf.ones_like(memory[:, 0, 0]), (-1, 1)) * 0.5 
        self.init_cumulated_alignments = tf.concat([tf.reshape(tf.ones_like(memory[:, 0, 0]), (-1, 1)), 
                                                    tf.zeros_like(memory[:, :, 0])[:, 1:]], axis=-1)

    def __call__(self, query, state):
        """Score the query based on the keys and values.
        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            state: cell_wrapper state
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        cumulated_alignments = state.cumulated_alignments 
        with variable_scope.variable_scope(None, "Location_Sensitive_Attention", [query]):

            # processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
            processed_query = self.query_layer(query) if self.query_layer else query
            # -> [batch_size, 1, attention_dim]
            processed_query = tf.expand_dims(processed_query, 1)

            # processed_location_features shape [batch_size, max_time, attention dimension]
            # [batch_size, max_time] -> [batch_size, max_time, 1]
            expanded_alignments = tf.expand_dims(cumulated_alignments, axis=2)
            # location features [batch_size, max_time, filters]
            f = self.location_convolution(expanded_alignments)
            # Projected location features [batch_size, max_time, attention_dim]
            processed_location_features = self.location_layer(f)

            # energy shape [batch_size, max_time]
            energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)

        # alignments shape = energy shape = [batch_size, max_time]
        previous_alignments = state.alignments 
        alignments = self._probability_fn(energy, previous_alignments) # has done masked padding

        # Cumulate alignments
        cumulated_alignments = state.cumulated_alignments + alignments 
        
        #forward attention 
        mu = state.mu 
        alpha = state.alpha # b * t  
        zeros = tf.zeros_like(self.keys[:, 0, 0])
        zeros = tf.reshape(zeros, (-1, 1))
        shift_alpha = tf.concat([zeros, alpha[:, :-1]], axis=-1)

        if not self.is_training and False:
            delta_val = -0.04
            mu = tf.clip_by_value(mu + delta_val, 0.0, 1.0)
        
        alignments = ((1 - mu) * alpha + mu * shift_alpha + 1e-10) * alignments 
        max_attentions = tf.argmax(alignments, -1, output_type=tf.int32) # (N, Ty/r)
        pos_rec = state.pos_rec # for saving time

        if not self.is_training and False: # prevent repeat and stay too long
            print('*' * 100)
            print('calling the part.')
            print('*' * 100)

            Tx = tf.shape(shift_alpha)[1]
            max_attentions = tf.where(tf.less_equal(max_attentions, state.max_attentions), 
                                            state.max_attentions, state.max_attentions+1)
            
            short_thres = tf.ones_like(state.pos_rec, dtype=tf.int32) * 5
            short_val = tf.ones_like(max_attentions) * 2 
            short_mask = tf.logical_and(tf.less(state.pos_rec, short_thres), 
                                        tf.less(short_val, max_attentions))

            max_attentions = tf.where(short_mask, state.max_attentions, max_attentions)

            pos_mask = tf.equal(max_attentions, state.max_attentions)
            ones_val = tf.ones_like(pos_mask, dtype=tf.int32)
            pos_rec = tf.where(pos_mask, state.pos_rec + 1, ones_val)
            
            thres = tf.ones_like(state.pos_rec, dtype=tf.int32) * 9
            pos_mask = tf.less(pos_rec, thres)

            max_attentions = tf.where(pos_mask, max_attentions, max_attentions+1)
            pos_rec = tf.where(pos_mask, pos_rec, ones_val)
            

            left = tf.sequence_mask(max_attentions-2, Tx)
            right = tf.logical_not(tf.sequence_mask(max_attentions+3, Tx))
                
            mask = tf.logical_not(tf.logical_or(left, right))
            paddings = tf.zeros_like(shift_alpha)
            alignments = tf.where(mask, alignments, paddings)
            
            left = tf.sequence_mask(tf.clip_by_value(max_attentions, 0, Tx-1), Tx)
            right = tf.logical_not(tf.sequence_mask(max_attentions+1, Tx))
            mask = tf.logical_not(tf.logical_or(left, right))

            max_alignments_values = tf.reduce_sum(alignments, axis=-1, keepdims=True)
            '''
            max_alignments_values = tf.where(tf.less(max_alignments_values, 
                                                tf.ones_like(max_alignments_values, dtype=tf.float32) * 1e-10),
                                            tf.ones_like(max_alignments_values, dtype=tf.float32), 
                                            max_alignments_values)
            '''

            alignments = tf.where(mask, tf.ones_like(alignments) * 1e-1 + max_alignments_values * 2.0, alignments)
            

        alignments = alignments / tf.reduce_sum(alignments, axis=-1, keepdims=True)
        expanded_alignments = tf.expand_dims(alignments, axis=1)
        context = math_ops.matmul(expanded_alignments, self.values)
        context = tf.squeeze(context, axis=1)

        '''
        if attention_layer:
            context = tf.layers.dense(tf.concat([context, query], axis=-1), units=query.shape[1])
        '''
        new_mu = tf.layers.dense(tf.concat([context, query], axis=-1), units=1, activation=tf.nn.sigmoid, use_bias=True)
        
        return alignments, new_mu, context, cumulated_alignments, max_attentions, pos_rec 


