# 利用指定说话人的少量数据微调预训练的Tacotron模型
---------------------------
**update at 2020-10-3**
----------------------------

> 由于master分支中采用开源语音数据集标贝(女声)，得到预训练TacotronV2模型，为了验证模型能够在指定说话人的少量音频以及对应的文本等数据下，微调预训练模型，特地从开源语音数据集thchs30中选择了D8(男声)，一共250句。下面的示例是根据D8来演示的，可以将D8改为自己的数据集。由于thchs30中所有的wav对应的文本中，没有标点符号，本人手动标注添加

## **[demo page](https://lturing.github.io/tacotronv2_wavernn_chinese/)**

**tensorflow-gpu的版本为1.14.0**
**参照[requirements.txt](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/requirements.txt)**安装相应的库 

## 测试(finetune)效果语音合成的效果  
> 以开源语音数据集thchs30中的说话人D8为例，微调 master分支中预训练的tacotron模型

```bash
git clone https://github.com/lturing/tacotronv2_wavernn_chinese.git
cd tacotronv2_wavernn_chinese
git checkout remotes/origin/adaptive　＃切换到adaptive分支
python tacotron_synthesize.py --text '现在是凌晨零点二十七分，帮您订好上午八点的闹钟。'
#合成的wav、attention align等在./tacotron_inference_output下
#由于在inference阶段，模型中的dropout没有关闭，相同的输入text，合成的wav的韵律等有轻微的不同
```

------------------------------


### TacotronV2训练数据集预处理
> 注意[hparams.py](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/adaptive/tacotron_hparams.py)中部分参数
```python
dataset = 'D8',
feat_out_dir = 'training_data',
tacotron_input = 'D8_train.txt',

tacotron_fine_tuning = True, 
pretrained_model_checkpoint_path = 'logs-Tacotron-2/taco_pretrained/tacotron_model.ckpt-206500',
pretrained_tacotron_input = 'biaobei_train.txt',

tacotron_initial_learning_rate = 1e-3, #starting learning rate

fmin = 55, #Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
fmax = 7600, #To be increased/reduced depending on data.

#M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
trim_fft_size = 2048, #Trimming window size
trim_hop_size = 512, #Trimmin hop length
trim_top_db = 22, #Trimming db difference from reference db (smaller==harder trim.)

```
### 参数补充说明
```
tacotron_fine_tuning　设为True
fmin 根据男声和女声取不同的值，男声(55)，女声(95)
trim_top_db 去掉音频首尾的静音部分，根据数据集，自行调整(对结果有影响)
```

> 执行如下脚本，生成TacotronV2的训练数据集(说话人D8)
```python
unzip D8.zip # 解压D8的原始数据(wav以及对应的文本)
python tacotron_preprocess.py
```

### 训练TacotronV2模型(默认finetune 3k步，loss在0.47左右)
> 执行如下脚本，训练TacotronV2模型(说话人D8)
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
