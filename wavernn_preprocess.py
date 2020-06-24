import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import wave
from datetime import datetime

import numpy as np
import tensorflow as tf
from tacotron.datasets import audio
from tacotron.utils.infolog import log
from librosa import effects
from tacotron.models import create_model
from tacotron.utils import plot
from tacotron.utils.text import text_to_sequence
import os
from tacotron_hparams import hparams
import shutil 
from tacotron.pinyin.parse_text_to_pyin import get_pyin


def padding_targets(target, r, padding_value):
    lens = target.shape[0]
    if lens % r == 0:
        return target 
    else:
        target = np.pad(target, [(0, r - lens % r), (0, 0)], mode='constant', constant_values=padding_value)
        return target 

class Synthesizer:
    def load(self, checkpoint_path, hparams, gta=False, model_name='Tacotron'):
        log('Constructing model: %s' % model_name)
        #Force the batch size to be known in order to use attention masking in batch synthesis
        inputs = tf.placeholder(tf.int32, (1, None), name='inputs')
        input_lengths = tf.placeholder(tf.int32, (1), name='input_lengths')

        targets = tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_targets')
        target_lengths = tf.placeholder(tf.int32, (1), name='target_length')
        gta = True 

        #initialize(self, inputs, input_lengths, mel_targets=None, stop_token_targets=None, 
        # linear_targets=None, targets_lengths=None, gta=False, global_step=None, is_training=False, 
        # is_evaluating=False)

        with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
            self.model = create_model(model_name, hparams)
            self.model.initialize(inputs=inputs, input_lengths=input_lengths, mel_targets=targets, 
            targets_lengths=target_lengths, gta=gta, is_evaluating=True)

            self.mel_outputs = self.model.mel_outputs
            self.alignments = self.model.alignments

        self._hparams = hparams

        self.inputs = inputs
        self.input_lengths = input_lengths
        self.targets = targets
        self.target_lengths = target_lengths 

        log('Loading checkpoint: %s' % checkpoint_path)
        #Memory allocation on the GPUs as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)


    def synthesize(self, text, mel, out_dir, idx):
        hparams = self._hparams
        r = hparams.outputs_per_step

        T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (0, hparams.max_abs_value)

        target = np.load(mel) 
        target = np.clip(target, T2_output_range[0], T2_output_range[1])
        target_length = target.shape[0]

        targets = padding_targets(target, r, T2_output_range[0])
        new_target_length = targets.shape[0]

        pyin, text = get_pyin(text)
        print(text)
        
        inputs = [np.asarray(text_to_sequence(pyin.split(' ')))]
        print(inputs)
        input_lengths = [len(inputs[0])]

        feed_dict = {
            self.inputs: np.asarray(inputs, dtype=np.int32),
            self.input_lengths: np.asarray(input_lengths, dtype=np.int32),
            self.targets: np.asarray([targets], dtype=np.float32),
            self.target_lengths: np.asarray([new_target_length], dtype=np.int32),
        }

        mels, alignments = self.session.run([self.mel_outputs, self.alignments], feed_dict=feed_dict)
        
        mel = mels[0]
        print('pred_mel.shape', mel.shape)
        mel = np.clip(mel, T2_output_range[0], T2_output_range[1])
        mel = mel[:target_length, :]
        mel = (mel + T2_output_range[1]) / (2 * T2_output_range[1]) 
        mel = np.clip(mel, 0.0, 1.0) # 0~1.0
        print(target_length, new_target_length)

        pred_mel_path = os.path.join(out_dir, 'mel-{}-pred.npy'.format(idx))
        np.save(pred_mel_path, mel, allow_pickle=False)
        plot.plot_spectrogram(mel, pred_mel_path.replace('.npy', '.png'), title='')

        alignment = alignments[0]
        alignment_path = os.path.join(out_dir, 'align-{}.png'.format(idx))
        plot.plot_alignment(alignment, alignment_path, title='')
        #alignment_path = os.path.join(out_dir, 'align-{}.npy'.format(idx))
        #np.save(alignment_path, alignment, allow_pickle=False) 
        

        return pred_mel_path, alignment_path


if __name__ == '__main__':
    synth = Synthesizer()
    cwd = os.getcwd()

    ckpt_path = os.path.join(cwd, 'logs-Tacotron-2/taco_pretrained')
    print(cwd, ckpt_path)
    checkpoint_path = tf.train.get_checkpoint_state(ckpt_path).model_checkpoint_path

    synth.load(checkpoint_path, hparams)
    print('succeed in loading checkpoint')
    
    out_dir = os.path.join(cwd, 'predicted_mel')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    base_path = '/home/spurs/tts/dataset/bznsyp/training_data_v1'

    cnt = 10
    res = open(os.path.join(cwd, 'wavernn_training_data.txt'), 'w', encoding='utf-8')
    with open(os.path.join(base_path, 'train.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            #audio_filename, mel_filename, time_steps, mel_frames, text, pyin
            line = line.strip().split('|')
            audio_name = line[0].strip() 
            wav_path = os.path.join(base_path, audio_name)
            wav = np.load(wav_path)
            wav = audio.encode_mu_law(wav)
            wav_path = os.path.join(out_dir, audio_name)
            np.save(wav_path, wav, allow_pickle=False)

            mel_path = os.path.join(base_path, line[1].strip())
            mel = np.load(mel_path)
            mel = (mel + hparams.max_abs_value) / ( 2 * hparams.max_abs_value)
            mel = np.clip(mel, 0, 1.0)
            mel_path_new = os.path.join(out_dir, line[1].strip())
            np.save(mel_path_new, mel, allow_pickle=False)

            text = line[-2].strip()
            idx = line[1].strip().split('-')[1].split('.')[0].strip() 
            print('idx=', idx)
            #break 

            pred_mel_path, alignment_path = synth.synthesize(text, mel_path, out_dir, idx)

            log = [wav_path, mel_path_new, pred_mel_path, text]

            res.write('|'.join(log) + '\n')
    
    res.close()
