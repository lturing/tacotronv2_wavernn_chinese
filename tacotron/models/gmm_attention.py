"""Attention file for location based attention (compatible with tensorflow attention wrapper)"""

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops, math_ops, nn_ops, variable_scope


class GMMAttention():

    def __init__(self,
                 memory,
                 memory_sequence_length,
                 is_training):

        self.memory = memory 
        self.memory_sequence_length = memory_sequence_length
        self.max_seq_len = tf.reduce_max(self.memory_sequence_length)
        self.is_training = is_training 
        self.batch_size = tf.shape(self.memory_sequence_length)[0]
        self.attention_size = 256 * 2 #tf.shape(memory)[-1]
        self.alignment_size = tf.shape(memory)[1]


    def _gmm_score(self, alpha, beta, kappa, num_attn_mixture):
        """Compute the window weights phi(t,u) of c_u at time t
        """
        
        u = tf.tile(
            tf.reshape(tf.range(self.max_seq_len), (1, 1, self.max_seq_len)), 
            (self.batch_size, num_attn_mixture, 1))
        
        u = tf.cast(u, tf.float32)
        alignments = tf.reduce_sum(alpha / beta * tf.exp(-tf.square(kappa - u) / beta), axis=1)

        return alignments 


    def __call__(self, query, state, num_attn_mixture):
        with variable_scope.variable_scope(None, "GMMAttention_forward", [query]):

            inputs = tf.concat([query, state.attention], axis=-1)
            inputs = tf.layers.dropout(inputs, rate=0.2, training=self.is_training)
            params = tf.layers.dense(inputs, units=3 * num_attn_mixture)
            alpha, beta, kappa = tf.split(tf.exp(params), 3, axis=1)
            kappa = kappa + state.kappa 
            alpha = tf.expand_dims(alpha, axis=-1)
            beta = tf.expand_dims(beta, axis=-1)
            kappa = tf.expand_dims(kappa, axis=-1)

            
            energy = self._gmm_score(alpha, beta, kappa, num_attn_mixture)
            #energy = tf.expand_dims(energy, axis=-1) # b * T * 1 

            Tx = self.max_seq_len
            mask = tf.sequence_mask(self.memory_sequence_length, maxlen=Tx)
            paddings = tf.ones_like(energy) * (-2 ** 32 + 1)  
            energy = tf.where(mask, energy, paddings) # B * T 
            alignments = tf.nn.softmax(energy) 
            alignments = tf.expand_dims(alignments, axis=-1)

            attentions = tf.reduce_sum(alignments * self.memory, axis=1)
            alignments = tf.squeeze(alignments, axis=-1)

            kappa = tf.squeeze(kappa, axis=-1)

        return alignments, attentions, kappa 
