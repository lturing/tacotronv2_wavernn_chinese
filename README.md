# tacotronv2_wavernn_chinese
> 利用开源中文语音数据集[标贝](https://www.data-baker.com/open_source.html)，训练中文[TacotronV2](https://github.com/Rayhane-mamah/Tacotron-2)，实现拼音输入序列到声学特征(Mel)转换的声学模型。在GTA模式下，利用训练好的TacotronV2合成标贝语音数据集中中文对应的Mel特征，作为声码器[WaveRNN](https://github.com/fatchord/WaveRNN)的训练数据。在合成阶段，利用TactornV2和WaveRNN合成高质量、高自然度的中文语音。
