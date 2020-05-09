# 启动服务
## request请求过程
1. 页面或客户端将需要合成的文字发送到Flask后台
2. Flask后台处理并转发到运行在docker里的Tensorflow Serving
3. Tensorflow Serving 合成语音的Mel谱，并返回给Flask
4. Flask接受Mel谱，并将其转化为语音，返回给页面或客户端

## 启动docker
```
docker run -p 8500:8500 \
  --mount type=bind,source=/home/spurs/tts/project/Tacotron-2_forward_attention/export,target=/models/tacotron_fw \
  -t --entrypoint=tensorflow_model_server tensorflow/serving \
  --port=8500\
  --model_name=tacotron_fw --model_base_path=/models/tacotron_fw &
```
其中
source=/home/spurs/tts/project/Tacotron-2_forward_attention/export #模型导出目录    
target=/models/tacotron_fw #docker内的目录(模型导出目录映射到到docker内的目录)    
--model_name=tacotron_fw #模型名称  
--model_base_path=/models/tacotron_fw  #docker内的进一步目录      



## 启动flask
```
python run.py 
```

## 页面
![avatar](/images/website.png)


## highlight
* 后台以base64的格式发送图片和wav，而不是url
![avatar](/images/post_result.png)



