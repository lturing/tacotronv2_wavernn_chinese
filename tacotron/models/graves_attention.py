#https://github.com/mozilla/TTS/blob/master/layers/common_layers.py

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops, math_ops, nn_ops, variable_scope
import numpy as np 


class GravesAttention():
    def __init__(self,
                 memory,
                 hparams,
                 num_atten,
                 is_training,
                 memory_sequence_length,
                 name='GravesAttention'):

        #Create normalization function
        #Setting it to None defaults in using softmax
        self.memory = memory 
        self.hparams = hparams 
        self.memory_sequence_length = memory_sequence_length
        self.max_sequence_len = tf.reduce_max(self.memory_sequence_length)
   
        self.eps = 1e-5 
        self.is_training = is_training 
        self.num_atten = num_atten 
        self.batch_size = tf.shape(memory)[0] 

        num_units = hparams.decoder_lstm_units // 4 
        #self.attention_layer_size = tf.shape(memory)[-1]
        self.attention_layer_size = memory.get_shape()[-1].value
        self.alignment_size = tf.shape(memory)[1]

        bias_init = tf.constant_initializer(np.hstack([np.zeros(self.num_atten),  
                                                      np.full(self.num_atten, 10), 
                                                      np.ones(self.num_atten)]))
                                                    
        layer1 = tf.layers.Dense(units=num_units, activation=tf.nn.relu, name="graves_attention_denselayer1", 
                                    trainable=True)

        layer2 = tf.layers.Dense(units=3*self.num_atten, bias_initializer=bias_init, name="graves_attention_denselayer2", 
                                trainable=True)

        self.dense_layer = lambda x: layer2(layer1(x))

        self.pos = tf.cast(tf.range(self.max_sequence_len + 1), dtype=tf.float32 ) + 0.5
        self.pos = tf.reshape(self.pos, (1, 1, -1))

        self.mask = tf.sequence_mask(self.memory_sequence_length, self.max_sequence_len)
        self.mask_value = 1e-20

        #self.paddings = tf.ones(shape, dtype=tf.float32) * self.mask_value
        self.paddings = tf.ones_like(memory[:, :, 0]) * self.mask_value 
    
    def splice_expand_dims(self, values, idx):
        # values : batch * num_attentions
        values = tf.expand_dims(values[:, idx], axis=1)
        return values 

    def __call__(self, query, state):

        seq_length = self.max_sequence_len 
        mu_prev = state.mu
        with variable_scope.variable_scope(None, "graves_attention", [query]):
            
            gbk_t = self.dense_layer(query)
            g_t, b_t, k_t = tf.split(gbk_t, num_or_size_splits=3, axis=1)

            mu_t = mu_prev + tf.math.softplus(k_t) # b * num_atten 
            #mu_t = mu_t + tf.ones_like(mu_t) * 0.05 
            sig_t = tf.math.softplus(b_t) + self.eps
            #g_t = tf.layers.dropout(g_t, rate=0.3, training=self.is_training)
            g_t = tf.nn.softmax(g_t, axis=1) + self.eps
            #x = (self.pos - tf.expand_dims(mu_t, -1)) / tf.expand_dims(sig_t, -1)
            #phi_t = tf.expand_dims(g_t, -1) * tf.nn.sigmoid(x)
            x = (tf.expand_dims(mu_t, -1) - self.pos) / tf.expand_dims(sig_t, -1)
            phi_t = tf.expand_dims(g_t, -1) * (1 / (1 + tf.nn.sigmoid(x)))
            # ref https://discourse.mozilla.org/t/graves-attention/51416
            #phi_t = tf.expand_dims(g_t, -1) * tf.exp(-0.5 * tf.expand_dims(sig_t, -1) *(self.pos - tf.expand_dims(mu_t, axis=-1)) ** 2)
            #phi_t = tf.expand_dims(g_t, -1) * tf.exp(-0.5 * (self.pos - tf.expand_dims(mu_t, -1)) ** 2 / tf.expand_dims(sig_t, axis=-1))

            alpha_t = tf.reduce_sum(phi_t, axis=1)
            alpha_t = alpha_t[:, 1:] - alpha_t[:, :-1]

            alpha_t = tf.where(self.mask, alpha_t, self.paddings) # alignment 

            max_attentions = tf.argmax(alpha_t, axis=-1, output_type=tf.int32)
            max_attentions_rec = state.max_attentions_rec
            
            if not self.is_training and False:
                mask = tf.less(state.max_attentions, max_attentions)
                new_rec = tf.where(mask, tf.ones_like(state.max_attentions_rec, dtype=tf.int32),
                                 state.max_attentions_rec+1)
                
                thres = tf.ones_like(new_rec) * 9
                mask = tf.less(new_rec, thres)
                mu_t = tf.where(mask, mu_t, mu_t+0.05)
                #max_attentions_rec = tf.where(mask, new_rec, tf.ones_like(new_rec, dtype=tf.int32))

                max_attentions_rec = new_rec 
            if not self.is_training and False:
                mu_t = mu_t - 0.2

            context = tf.reduce_sum(tf.expand_dims(alpha_t, axis=-1) * self.memory, axis=1) 

            #if False: context = tf.layers.dense(tf.concat([context, query], axis=-1), units=tf.shape(context)[1])

        return alpha_t, mu_t, max_attentions, max_attentions_rec, context 

