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
        plot.plot_spectrogram(mel, pred_mel_path, title='')
        
        #alignment_path = os.path.join(out_dir, 'step-{}-{}-align.npy'.format(step, idx))
        #np.save(alignment_path, alignment, allow_pickle=False) 
        alignment_path = os.path.join(out_dir, 'step-{}-{}-align.png'.format(step, idx))
        plot.plot_alignment(alignment, alignment_path,
            title='{}'.format(''), split_title=True, max_len=target_length)

        return pred_mel_path, alignment_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default='', help='text to synthesis.')   
    args = parser.parse_args()

    past = time.time()

    synth = Synthesizer()

    ckpt_path = os.path.join(cwd, 'logs-Tacotron-2/taco_pretrained')
    checkpoint_path = tf.train.get_checkpoint_state(ckpt_path).model_checkpoint_path

    synth.load(checkpoint_path, hparams)
    print('succeed in loading checkpoint')
    
    out_dir = os.path.join(cwd, 'tacotron_inference_output')
    #if os.path.exists(out_dir):
    #    shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    #text = '分析国内外新冠肺炎疫情防控形势，研究部署抓紧抓实抓细常态化疫情防控工作；分析研究当前经济形势，部署当前经济工作。中共中央总书记习近平主持会议。'
    #text = '分析国内外新冠肺炎疫情防控形势，研究部署抓紧抓实抓细常态化疫情防控工作；分析研究当前经济形势，部署当前经济工作。中共中央总书记习近平主持会议。中共中央总书记国家主席中央军委主席习近平4月8日给武汉市东湖新城社区全体社区工作者回信，再次肯定城乡广大社区工作者在疫情防控斗争中发挥的重要作用，向他们致以诚挚的慰问，并勉励他们为彻底打赢疫情防控人民战争、总体战、阻击战再立新功。习近平在回信中说，我从武汉回来后，一直牵挂着武汉广大干部群众，包括你们社区在内的武汉各社区生活正在逐步恢复正常，我感到很高兴。习近平指出，在这场前所未有的疫情防控斗争中，城乡广大社区工作者同参与社区防控的各方面人员一道，不惧风险、团结奋战，特别是社区广大党员、干部以身作则、冲锋在前，形成了联防联控、群防群控的强大力量，充分彰显了打赢疫情防控人民战争的伟力。习近平强调，现在，武汉已经解除了离汉离鄂通道管控措施，但防控任务不可松懈。社区仍然是外防输入、内防反弹的重要防线，关键是要抓好新形势下防控常态化工作。希望你们发扬连续作战作风，抓细抓实疫情防控各项工作，用心用情为群众服务，为彻底打赢疫情防控人民战争、总体战、阻击战再立新功。社区是疫情防控的最前线。新冠肺炎疫情发生以来，全国400多万名社区工作者坚守一线，在65万个城乡社区从事着疫情监测、出入管理、宣传教育、环境整治、困难帮扶等工作，为遏制疫情扩散蔓延、保障群众生活作出了重要贡献。习近平总书记高度重视社区防控工作，多次作出重要指示，对社区工作者给予肯定，还先后到北京市安华里社区、武汉市东湖新城社区考察慰问。近日，武汉市东湖新城社区的全体社区工作者给习总书记写信，表达了对总书记和党中央的感激之情，以及继续坚守好阵地、履行好职责的坚强决心。'
    
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

    text = '华为积极开展与产业界、开发者、学术界、产业标准组织的密切合作，推动商业和科技创新，推动业界建立合作共赢、公平竞争的产业健康发展生态。从华为上我们能看到什么？除了能看到华为对于影像的追求，更重要的是华为对年轻人的理解是否合理，能否打动更多的年轻人。为什么要这么强调年轻人呢？因为年轻人代表的就是未来。华为要想成长的更好，就必须抓住未来。平面几何问题有的时候可以使用解析几何的语言来描述，但是直接设点用解析几何语言描述关系有的时候计算会非常繁杂。近未来的地球黄沙遍野，小麦、秋葵等基础农作物相继因枯萎病灭绝，人类不再像从前那样仰望星空，放纵想象力和灵感的迸发，而是每日在沙尘暴的肆虐下倒数着所剩不多的光景。在家务农的前美国国家航空航天局宇航员库珀（马修·麦康纳 饰）接连在女儿墨菲（麦肯吉·弗依 饰）的书房发现奇怪的重力场现象，随即得知在某个未知区域内前美国国家航空航天局成员仍秘密进行一个拯救人类的计划。多年以前土星附近出现神秘虫洞，美国国家航空航天局借机将数名宇航员派遣到遥远的星系寻找适合居住的星球。在布兰德教授（迈克尔·凯恩 饰）的劝说下，库珀忍痛告别了女儿，和其他三名专家教授女儿艾米莉亚·布兰德（安妮·海瑟薇 饰）、罗米利（大卫·吉雅西 饰）、多伊尔（韦斯·本特利 饰）搭乘宇宙飞船前往目前已知的最有希望的三颗星球考察。他们穿越遥远的星系银河，感受了一小时七年光阴的沧海桑田，窥见了未知星球和黑洞的壮伟与神秘。在浩瀚宇宙的绝望而孤独角落，总有一份超越了时空的笃定情怀将他们紧紧相连。《星际穿越》里面最容易被忽略、最令人难过的角色，一个重要的角色。应该去看第二遍，从头看一下他的戏。那个人是墨菲的哥哥。一个普通的、不会和妹妹沟通的哥哥，一个立志当农夫的少年。和墨菲不一样。墨菲觉得当宇航员的爸爸很棒，但汤姆一直觉得农夫爸爸才是最酷的。看棒球赛的时候，他亲口确认“我喜欢你做的那些事儿。” 爸爸离开前， 他试着让爸爸给他一个承诺：留下他的车。后来美国国家航空航天局的人果然把爸爸的车开回来了。爸爸答应了就不会食言，他就这样相信着。这是导演安插的，一个小小的细节。他孤独地和妻子，小孩生活在农场，本来打算就这样终老一生。一个心智正常的人，为什么要把自己关在一个气候恶劣的小镇而拒绝迁徙呢？他只是个顽固的农夫吗？他不是可有可无的角色，诺兰不会让这事儿发生。事实上，只有他始终相信爸爸会回来。 别忘了，过去的二十三年里，每年给爸爸发视频信息的都会是他。他可以是我们身边的任何人，那些我们认为不会改变世界的小人物。在英雄们改变世界的时候，那些“被牺牲”的人不是墨菲，也不是基地里的科学家，而是更普通的人。但这些人拥有的信念，胜过那些“被选中”去改变世界的人。就我所知的地质学知识而言，这种星球可能不太合理。因为这种环境下，常见的岩石都会被破碎，剥蚀，和水混合，像泥浆一样四处翻涌。因为长期处于浪基面以上的剥蚀环境中，海底表层应该是松软的沉积物，巨浪袭来应该会裹挟大量砂质，而浑浊不堪，而不是电影里那种清澈海水。在这种情况下，望远镜的分辨率取决于望远镜之间的距离，而非单个望远镜口径的大小，所以，视界面望远镜的分辨率相当于一部口径为地球直径大小的射电望远镜的分辨率。为了增加空间分辨率，以看清更为细小的区域，科学家们在此次进行观测的望远镜阵列里增加了位于智利和南极的望远镜。要保证所有八个望远镜都能看到这两个黑洞，从而达到最高的灵敏度和最大的空间分辨率，留给科学家们的观测窗口期非常短暂，每年只有大约十天时间。全都是纸屑，全部要改写。对敌人谦卑，抱歉我不会，而远方龙战于野，咆哮声不自觉，横越过了几条街。我坚决，冲破这一场浩劫，这世界谁被狩猎，谁淌血我却只为，拯救你的无邪。城墙上我在等魔坠，火焰吞噬无名碑，摧毁却无法击溃，我要爱上谁。废墟怎么被飞雪了解，只能滋长出羊齿蕨，那些仇恨已形成堡垒，我又该怎么去化解。低吼威胁那些龙形的傀儡，它们发不出的音叫心碎，惊觉你啜泣声迂回。如此纯洁，以温柔削铁，以爱在谅解，在末日边陲，纯爱被隔绝，我在危险的交界，目睹你的一切，锈迹斑斑的眼泪。我坚决冲破这一场浩劫。这世界谁被狩猎，谁淌血我却只为拯救你的无邪，城墙上我在等魔坠，火焰吞噬无名碑，摧毁却无法击溃。我要爱上谁，我坚决冲破这一场浩劫。这世界谁被狩猎，摧毁却无法击溃，我要爱上谁。大热美剧《西部世界》在全球拥有无数粉丝，烧脑的剧情、宏伟的世界观、复仇的人工智能。无一不让大家对神秘的西部世界充满向往。不过我们今天要说的西部世界，不是那个神秘莫测的科幻世界。而是位于美国西部中心的内华达州。杜兰特受伤之后接受采访时，说自己和威少是一种真正的兄弟关系，他们就像一家人一样。他无法接受外界传言的自己不喜欢威少，他重申自己喜欢并尊重威少，因为他们有着相同的目标和职业素养。杜兰特与威少爷，曾像亲兄弟一样互相扶持，他们几乎踏上世界之巅，也一度经历最让人遗憾的失败。在他们最巅峰的那几年里，实在是太多故事可以讲了。杜兰特断掉了蝉联了的三年得分王，威少爷仍有短板，但他的长板已然卓越，杜兰特索性放权，让威少去主持进攻，这让他们更加融洽。季后赛威少受伤只打了两场比赛，后面的路就不太好走。遇到灰熊，黑白双煞像两台绞肉机。雷霆没有了最好的提速器，那个赛季戛然而止。'

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
    text = '您好，我这边是中邮消费这边的客服，请问有什么我可以帮助到您？'

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

    print('last: {} seconds'.format(time.time() - past))

