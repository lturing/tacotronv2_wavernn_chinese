# TacotronV2 + WaveRNN

---------------------------------
**update at 2020-10-3 添加微调分支[adaptive](https://github.com/lturing/tacotronv2_wavernn_chinese/tree/adaptive)**
---------------------------------

1. 开源中文语音数据集[标贝](https://www.data-baker.com/open_source.html)(女声)训练中文[TacotronV2](https://github.com/Rayhane-mamah/Tacotron-2)，实现中文到声学特征(Mel)转换的声学模型。在GTA模式下，利用训练好的TacotronV2合成标贝语音数据集中中文对应的Mel特征，作为声码器[WaveRNN](https://github.com/fatchord/WaveRNN)的训练数据。在合成阶段，利用TactornV2和WaveRNN合成高质量、高自然度的中文语音。
2. 从[THCHS-30](http://www.openslr.org/18/)任选一个speaker的语音数据集，微调TacotronV2中的部分参数，实现说话人转换[branch adaptive](https://github.com/lturing/tacotronv2_wavernn_chinese/tree/adaptive)。
3. Tensorflow serving + Flask 部署TacotronV2中文语音合成服务。   

由于[TacotronV2](https://github.com/Rayhane-mamah/Tacotron-2)[TacotronV2](https://github.com/mozilla/TTS)中采用Location sensitive attention，对长句字的建模能力不好(漏读、重复)，尝试了[GMM attention](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/models/gmm_attention.py)、[Discrete Graves Attention](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/models/graves_attention.py)[issue](https://github.com/mozilla/TTS/issues/346)、[Forward attention](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/models/forward_attention.py)，能有效地解决对长句的建模能力，加快模型收敛速度。

## **[demo page](https://lturing.github.io/tacotronv2_wavernn_chinese/)**

**tensorflow-gpu的版本为1.14.0**

## 测试语音合成的效果       
**参照[requirements.txt](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/requirements.txt)**安装相应的库      
```bash
git clone https://github.com/lturing/tacotronv2_wavernn_chinese.git
cd tacotronv2_wavernn_chinese
python tacotron_synthesize.py --text '现在是凌晨零点二十七分，帮您订好上午八点的闹钟。'
#合成的wav、attention align等在./tacotron_inference_output下
#由于在inference阶段，模型中的dropout没有关闭，相同的输入text，合成的wav的韵律等有轻微的不同
```
------------------------------

## 训练TacotronV2模型

### 训练数据集预处理
#### 中文标点符号处理
对于中文标点符号，只保留'，。？！'四种符号，其余符号按照相应规则转换到这四个符号之一。

#### 中文到拼音转换[code](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/pinyin/parse_text_to_pyin.py)
利用[字拼音文件](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/pinyin/pinyin.txt)和[词拼音文件](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/pinyin/large_pinyin.txt)，实现中文到拼音转换，能有效消除多音字的干扰。具体步骤如下：       
1. 对于每个句子中汉字从左到右的顺序，优先从词拼音库中查找是否存在以该汉字开头的词并检查该汉字后面的汉字是否与该词匹配，若满足条件，直接从词库中获取拼音，若不满足条件，从字拼音库中获取该汉字的拼音。     
2. 对于数字(整数、小数)、ip地址等，首先根据规则转化成文字，比如整数2345转化为二千三百四十五，再转化为拼音。
3. 由于输入是文字转化而来的拼音序列，所以在合成阶段，允许部分或全部的拼音输入。    

*注优先从词文件中寻找拼音，也会带来错误的拼音(也许先分词能解决)，所以本项目支持中文和拼音混合输入*


### TacotronV2训练数据集预处理
> 修改[hparams.py](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron_hparams.py)中[标贝数据集](https://www.data-baker.com/open_source.html)的路径
```python
dataset = '/home/spurs/tts/dataset/bznsyp', #标贝数据集的根目录，其wav文件在 dataset/bznsyp下
base_dir = '/home/spurs/tts/dataset',
feat_out_dir = 'training_data_v1',
tacotron_input = '/home/spurs/tts/dataset/bznsyp/training_data_v1/train.txt', 
```

> 执行如下脚本，生成TacotronV2的训练数据集
```python
python tacotron_preprocess.py
```

### 训练TacotronV2模型
> 执行如下脚本，训练TacotronV2模型
```python
python tacotron_train.py
```

### TacotronV2合成Mel频谱
> TacotronV2生成Mel文件，利用griffin lim算法恢复语音，修改脚本 tacotron_synthesize.py 中text
```python
python tacotron_synthesize.py
```

或命令行输入   
```python
python tacotron_synthesize.py --text '国内知名的视频弹幕网站，这里有最及时的动漫新番。'
```

**[TacotronV2 pretrained model](https://github.com/lturing/tacotronv2_wavernn_chinese/tree/master/logs-Tacotron-2/taco_pretrained)**


### 改进部分
> 由于[TacotornV2]()中采用的注意力机制是[Location sensitive attention](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/models/location_sensitive_attention.py)，对长句子的建模能力不太好，尝试了以下注意力机制：    
* [Guassian mixture attention](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/models/gmm_attention.py)
* [Discretized Graves attention](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/models/graves_attention.py)
* [Forward attention](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/models/forward_attention.py)

> 由于语音合成中的音素(拼音)到声学参数(Mel频谱)是从左到右的单调递增的对应关系，特别地，在合成阶段，对[forward attention](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/models/forward_attention.py#L171)中的alignments的计算过程的特殊处理，能进一步提高模型对长句子的语音合成效果，以及控制语速。

## 说话人转换(speaker adaptive)
[TactronV2](https://github.com/Rayhane-mamah/Tacotron-2)支持finetune，固定decoder层前的参数(embedding层、CHBG、encoder层等)，用新数据集(数据量很少)训练从checkpoint中恢复的模型，达到speaker adpative的目的。


## 训练WaveRNN模型
### Wavernn训练数据集准备
> 利用训练好的TacotronV2对标贝语音数据集在GTA(global teacher alignment)模式下，生成对应的Mel特征。需要注意的如下：    
* TacotronV2中的mel输出的范围为[-hparmas.max_abs_value, hparams.max_abs_value]，而WaveRNN中的mel的范围[0, 1]，故需要将TacotronV2输出mel特征变为[0, 1]范围内。
* TacotronV2中的hop_size为275，需要将WaveRNN中的voc_upsample_factors的值改为(5, 5, 11)(注上采样的比例, 或者(x, y, z)，并且x * y * z = hop_size)。
* Wavernn中voc_mode设为RAW，bits为10，故需要将[wav文件](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/datasets/audio.py#L8)转换到相应格式。

```python
python wavernn_preprocess.py #利用训练好的TacotronV2生成Wavernn的训练数据
```

### Wavernn模型训练
> 训练前需要切换到pytorch conda环境，例如：    
```shell
conda activate torch1.0 #切换到pytorch分支
```
> 训练模型
```python
python wavernn_train.py
```

### Wavernn模型评估
```python
python wavernn_gen.py --file path_to_mel_generated_by_tacotronv2 
```

**[Wavernn pretrained model](https://github.com/lturing/tacotronv2_wavernn_chinese/tree/master/logs_wavernn/checkpoints)**

## 服务部署 
**[website](https://github.com/lturing/tacotronv2_wavernn_chinese/tree/master/website)**     
采用Tensorflow Serving + Docker 来部署训练好的TacotronV2语音服务，由于需要对文本进行处理，还搭建了Flask后台框架，最终的语音合成的请求过程如下：       
请求过程：页面 -> Flask后台 -> Tensorflow serving    
响应过程：Tensorflow serving -> Flask后台 -> 页面


## 额外参照文献
- [location-relative attention mechanisms for robust long-form speech synthesis](https://arxiv.org/pdf/1910.10288)
