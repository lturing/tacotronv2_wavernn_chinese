from app import app
from flask import request, render_template, jsonify
import json
import base64
import io
from scipy.io.wavfile import write
from app.audio import inv_mel_spectrogram, save_wav
from app.text_to_pyin import get_pyin
from app.text import text_to_sequence
from app.plot import plot_alignment
import requests
import numpy as np 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time 

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf  
import numpy as np
from grpc.beta import implementations
import grpc

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

def request_server(seq, seq_length, host, port):
    options = [('grpc.max_send_message_length', 1000 * 1024 * 1024), 
            ('grpc.max_receive_message_length', 1000 * 1024 * 1024)]  

    #channel = implementations.insecure_channel(host, port=port, options=options)
    channel = grpc.insecure_channel(target="{}:{}".format(host, port), options=options)
    #stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = "tacotron_fw"  # 模型名称
    request.model_spec.signature_name = "tacotron_fw"  # 签名名称

    # 导出模型时设置的输入名称
    request.inputs["input"].CopyFrom(
        tf.contrib.util.make_tensor_proto(seq, shape=seq.shape ))
    request.inputs['input_length'].CopyFrom(
        tf.contrib.util.make_tensor_proto(seq_length, shape=[1]))

    response = stub.Predict.future(request, 200.0)  # 5 secs timeout
    return response.result()
    #return response.outputs['mel'], reponse.outputs['alignment']

@app.route('/generate_tts', methods=['POST'])
def generate_tts():
    txt = request.form.get('txt')
    ret = {}
    ret['txt'] = txt
    pyin, txt = get_pyin(txt)
    #print('pyin=', pyin)

    seq = np.asarray(text_to_sequence(pyin), dtype=np.int32)
    #print('seq=', seq.tolist())
    
    seq_length = np.asarray([len(seq)], dtype=np.int32)
    seq = np.reshape(seq, [1, len(seq)])

    host = '0.0.0.0'
    port = 8500
    
    '''
    data = '{{"input": {}, "input_length": {}}}'.format(str(seq), str(seq_length))
    #response = requests.post('http://0.0.0.0:8500/v1/models/tacotron_fw:predict', data=data)
    #data = json.dumps({'input': seq, 'input_length': seq_length})
    data = {'input': seq, 'input_length': seq_length}
    response = requests.post('http://localhost:8500/v1/models/tacoton_fw:predict', json=data)
    #response = requests.post('http://localhost:8500/v1/models/tacoton_fw:predict', data=data)
    '''
    response = request_server(seq, seq_length, host, port)
    #mel = response.json()['mel'][0]
    mel = tf.make_ndarray(response.outputs["mel"])
    wav = inv_mel_spectrogram(mel.T)
    wav = save_wav(wav)
    print('wav.shape', wav.shape)
    
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    rate=22050
    write(byte_io, rate, wav)
    
    #ref https://stackoverflow.com/questions/59338606/send-audio-data-represent-as-numpy-array-from-python-to-javascript
    #ref https://gist.github.com/hadware/8882b980907901426266cb07bfbfcd20
    wav = base64.b64encode(byte_io.read()).decode('utf-8')
    byte_io.close()
    wav = "data:audio/wav;base64, %s" % wav
    ret['wav'] = wav
    ret['pyin'] = pyin

    #align = response.json()['alignment'][0]
    align = tf.make_ndarray(response.outputs["alignment"])
    img_buff = plot_alignment(align)
    img = "data:image/jpeg;base64, %s" % base64.b64encode(img_buff.read()).decode('utf-8')
    img_buff.close()
    ret['img'] = img
    return jsonify(ret)
    #return json.dumps(ret)
