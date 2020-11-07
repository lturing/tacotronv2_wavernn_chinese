import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""

cwd = os.getcwd()

import sys
sys.path.append(cwd)

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
import hashlib 
import time 
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

        targets = None #tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_targets')
        target_lengths = None #tf.placeholder(tf.int32, (1), name='target_length')
        #gta = True 

        with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
            self.model = create_model(model_name, hparams)
            self.model.initialize(inputs=inputs, input_lengths=input_lengths)
            #mel_targets=targets,  targets_lengths=target_lengths, gta=gta, is_evaluating=True)

            self.mel_outputs = self.model.mel_outputs
            self.alignments = self.model.alignments
            if hparams.predict_linear:
                self.linear_outputs = self.model.linear_outputs
            self.stop_token_prediction = self.model.stop_token_prediction

        self._hparams = hparams

        self.inputs = inputs
        self.input_lengths = input_lengths
        #self.targets = targets
        #self.target_lengths = target_lengths 

        log('Loading checkpoint: %s' % checkpoint_path)
        #Memory allocation on the GPUs as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)


    def synthesize(self, text, out_dir, idx, step):
        hparams = self._hparams

        T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (0, hparams.max_abs_value)

        #pyin, text = get_pyin(text)
        print(text.split(' '))
        
        inputs = [np.asarray(text_to_sequence(text.split(' ')))]
        print(inputs)
        input_lengths = [len(inputs[0])]

        feed_dict = {
            self.inputs: np.asarray(inputs, dtype=np.int32),
            self.input_lengths: np.asarray(input_lengths, dtype=np.int32),
        }
        
        mels, alignments, stop_tokens = self.session.run([self.mel_outputs,
                self.alignments, self.stop_token_prediction], feed_dict=feed_dict)

        mel = mels[0]
        alignment = alignments[0]

        print('pred_mel.shape', mel.shape)
        stop_token = np.round(stop_tokens[0]).tolist()
        target_length = stop_token.index(1) if 1 in stop_token else len(stop_token)

        mel = mel[:target_length, :]
        mel = np.clip(mel, T2_output_range[0], T2_output_range[1])

        wav_path = os.path.join(out_dir, 'step-{}-{}-wav-from-mel.wav'.format(step, idx))
        wav = audio.inv_mel_spectrogram(mel.T, hparams)
        audio.save_wav(wav, wav_path, sr=hparams.sample_rate)
        
        pred_mel_path = os.path.join(out_dir, 'step-{}-{}-mel-pred.npy'.format(step, idx))
        new_mel = np.clip((mel + T2_output_range[1]) / (2 * T2_output_range[1]), 0, 1)
        np.save(pred_mel_path, new_mel, allow_pickle=False)

        pred_mel_path = os.path.join(out_dir, 'step-{}-{}-mel-pred.png'.format(step, idx))
        plot.plot_spectrogram(mel, pred_mel_path, title=datetime.now().strftime('%Y-%m-%d %H:%M'))
        
        #alignment_path = os.path.join(out_dir, 'step-{}-{}-align.npy'.format(step, idx))
        #np.save(alignment_path, alignment, allow_pickle=False) 
        alignment_path = os.path.join(out_dir, 'step-{}-{}-align.png'.format(step, idx))
        plot.plot_alignment(alignment, alignment_path,
            title=datetime.now().strftime('%Y-%m-%d %H:%M'), split_title=True, max_len=target_length)

        return pred_mel_path, alignment_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default='', help='text to synthesis.')   
    args = parser.parse_args()

    past = time.time()

    synth = Synthesizer()

    ckpt_path = 'logs-Tacotron-2/taco_pretrained'
    checkpoint_path = tf.train.get_checkpoint_state(ckpt_path).model_checkpoint_path

    synth.load(checkpoint_path, hparams)
    print('succeed in loading checkpoint')
    
    out_dir = os.path.join(cwd, 'tacotron_inference_output')
    #if os.path.exists(out_dir):
    #    shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    #text = '分析国内外新冠肺炎疫情防控形势，研究部署抓紧抓实抓细常态化疫情防控工作；分析研究当前经济形势，部署当前经济工作。中共中央总书记习近平主持会议。'
    
    #text = '中共中央总书记，国家主席，中央军委主席习近平4月8日给武汉市东湖新城社区全体社区工作者回信，再次肯定城乡广大社区工作者在疫情防控斗争中发挥的重要作用。'
    #text = '对敌人谦卑，抱歉我不会，而远方龙战于野。'
    #text = '不好意思，您能再说一遍吗？'
    #text = '不好意思，您能再说一遍吗。'
    #text = '平面几何问题有的时候可以使用解析几何的语言来描述，但是直接设点用解析几何语言描述关系有的时候计算会非常繁杂。'
    #text = '近未来的地球黄沙遍野，小麦秋葵等基础农作物相继因枯萎病灭绝，人类不再像从前那样仰望星空，放纵想象力和灵感的迸发，而是每日在沙尘暴的肆虐下倒数着所剩不多的光景。'
    #text = '我们来比谁知道的水果多，你先说一个水果的名字吧，没听清可以说重复。'
    text = '哈尔滨今天晴，十度到二十二度，南风三级，空气质量良。'
    #text = '现在是凌晨零点二十七分，帮您订好上午八点的闹钟。'
    #text = '好啊，一起来听张学友的我等得花儿也谢了。'
    #text = '好啊？一起来听张学友的我等得花儿也谢了。'
    #text = '好啊！一起来听张学友的我等得花儿也谢了。'

    #text = '据德国《西部日报》二十二日报道，荷兰北部弗里斯兰省一些地区，最近小龙虾泛滥成灾。这些小家伙在水里大量繁殖，还挥舞着钳子走上了街道导致当地居民甚至无法正常出门散步。'
    
    #text = '近未来的地球黄沙遍野，小麦秋葵等基础农作物相继因枯萎病灭绝，人类不再像从前那样仰望星空，放纵想象力和灵感的迸发，而是每日在沙尘暴的肆虐下倒数着所剩不多的光景。'
    #text = '给予您给予您给予您给予您给予您给予您给予您给予您。'
    #text = '数星星的工作让科学家发现了天体在宇宙里的分布和运动规律，这也是最早的天文学研究方法。那天上的星星是什么？它的物理本质是什么？起源是什么？内部结构是什么？又如何演化？最终命运又是什么？这些疑问激起了物理学家的极大兴趣。'

    #text = '在家务农的前美国国家航空航天局宇航员库珀马修·麦康纳饰接连在女儿墨菲麦肯吉·弗依饰的书房发现奇怪的重力场现象。'

    #text = '如果打穿地球，那么从一头到另一头h ui4发生什么？'

    #text = '女儿，女儿，女儿，' * 10 + '。'
    #text = '我点燃那盏灯火，向远方凝望着，空气都打开了。记忆随风散落，幻想美好的时刻，没有完美结果。红色夕阳下落，黯淡的云朵，憧憬像飘浮的泡沫，光映出灿烂的颜色，可却没有照到我，全世界的雨打到我，我的梦早已湿透了，瞬间被淹没。我点燃那盏灯火，向远方凝望着，空气都打开了。'
    
    #text = '现在是凌晨零点二十七分，帮您订好上午八点的闹钟。'
    #text = '这是一个人与人之间无比接近的时代，近到，拿起手机，你可以和世界上任何一个角落的人，无缝地交流，连接。这是一个人与人之间无比遥远的时代，远到，即使你身边坐满了人， 也未必有人愿意听你说一句心里话。你的孤独没人懂。这个时候，越来越多的人干脆，选择抛弃同类，转身去和人工智能谈情说爱，做朋友。'
   
    #text = '您好，麻烦您帮我拿一下我的书包。'

    #text = '零，一，二，三，四，五，六，七，八，九，十。'

    #text = '中邮消费金融来电是想提醒您，您的贷款已逾期，如果业务上有问题请致电客服四零零，六六九五，五八零，再见！'

    text = args.text if args.text != '' else text 
    pyin, text = get_pyin(text)
    
    m = hashlib.md5()
    m.update(text.encode('utf-8'))
    idx = m.hexdigest()
    step = checkpoint_path.split('/')[-1].split('-')[-1].strip()

    #mel_path = os.path.join(out_dir, idx+'_mel.npy')
    pred_mel_path, alignment_path = synth.synthesize(pyin, out_dir, idx, step)
    print(text)
    print(checkpoint_path)
    print(idx)

    print('last: {} seconds'.format(time.time() - past))

