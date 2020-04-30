import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time 

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import tensorflow as tf  
import numpy as np
from grpc.beta import implementations


def request_server(seq, host, port):

    channel = implementations.insecure_channel(host, port=port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "tacotron_fw"  # 模型名称
    request.model_spec.signature_name = "tacotron_fw"  # 签名名称

    # 导出模型时设置的输入名称
    request.inputs["input"].CopyFrom(
        tf.contrib.util.make_tensor_proto(seq, shape=[1, len(seq)] ))
    request.inputs['input_length'].CopyFrom(
        tf.contrib.util.make_tensor_proto([len(seq)], shape=[1]))

    response = stub.Predict(request, 200.0)  # 5 secs timeout
    return reponse
    #return response.outputs['mel'], reponse.outputs['alignment']

