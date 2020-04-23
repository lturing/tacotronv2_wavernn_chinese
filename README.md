# tacotronv2_wavernn_chinese
1. 利用开源中文语音数据集[标贝](https://www.data-baker.com/open_source.html)(女声)，训练中文[TacotronV2](https://github.com/Rayhane-mamah/Tacotron-2)，实现拼音输入序列到声学特征(Mel)转换的声学模型。在GTA模式下，利用训练好的TacotronV2合成标贝语音数据集中中文对应的Mel特征，作为声码器[WaveRNN](https://github.com/fatchord/WaveRNN)的训练数据。在合成阶段，利用TactornV2和WaveRNN合成高质量、高自然度的中文语音。
2. 从[THCHS-30](http://www.openslr.org/18/)任选一个speaker，finetune TacotronV2中的部分参数，实现speaker　adaptive。

## 1 训练中文Tacotron V2

### 1.1 训练数据集预处理
#### 1.1.1 中文标点符号处理
对于中文标点符号，只保留'，。？！'四种符号，其余符号按照相应规则转换到这四个符号之一。

#### 1.1.2 中文到拼音转换
利用python读字拼音库文件和词拼音库文件，实现中文到拼音转换，能有效消除多音字的干扰。具体步骤如下：
对于每个句子中汉字从左到右的顺序，优先从词拼音库中查找是否存在以该汉字开头的词并检查该汉字后面的汉字是否与词匹配，若满足条件，直接从词库中获取拼音，若不满足条件，从字拼音库中获取该汉字的拼音。
并且，对于数字(整数、小数)、ip地址等，首先根据规则转化成文字，比如整数2345转化为二千三百四十五，再转化为拼音

###　1.2 训练数据集预处理
将[hparams.py](https://github.com/Rayhane-mamah/Tacotron-2/blob/master/hparams.py)中参数修改如下：
```python
num_freq = 513
n_fft = 1024
hop_size = 256
win_size = 1024 # 改成与n_fft一样，为了后续训练SqueezeWave
sample_rate = 22050
trim_silence = True # 去掉音频中开头和结尾处的静音部分
trim_fft_size = 512
trim_hop_size = 128
trim_top_db = 40
fmin = 95
outputs_per_step = 1 # outputs_per_step越大，收敛越快(attention 对齐)，但效果有一定程度变差
predict_linear = False # 
```
同时，由于GTX 1060 6G的gpu的限制，拼音embedding size，encoder 和 decoder中的lstm的单元数有所减少，具体参数参看
执行如下脚本，生成TacotronV2的训练数据集
> python preprocess.py

### 1.2 训练模型

 
