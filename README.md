# tacotronv2_wavernn_chinese
利用开源中文语音数据集[标贝](https://www.data-baker.com/open_source.html)，训练中文[TacotronV2](https://github.com/Rayhane-mamah/Tacotron-2)，实现拼音输入序列到声学特征(Mel)转换的声学模型。在GTA模式下，利用训练好的TacotronV2合成标贝语音数据集中中文对应的Mel特征，作为声码器[WaveRNN](https://github.com/fatchord/WaveRNN)的训练数据。在合成阶段，利用TactornV2和WaveRNN合成高质量、高自然度的中文语音。

## 训练中文Tacotron V2

### 中文预处理
#### 标点符号处理
对于中文标点符号，只保留'，。？！'四种符号，其余符号按照相应规则转换到这四个符号之一。

#### 中文到拼音转换
利用python读字拼音库文件和词拼音库文件，实现中文到拼音转换，能有效消除多音字的干扰。具体步骤如下：
对于每个句子中汉字从左到右的顺序，优先从词拼音库中查找是否存在以该汉字开头的词并检查该汉字后面的汉字是否与词匹配，若满足条件，直接从词库中获取拼音，若不满足条件，从字拼音库中获取该汉字的拼音。
并且，对于数字(整数、小数)、ip地址等，首先根据规则转化成文字，比如整数2345转化为二千三百四十五，再转化为拼音

#### 中文

 