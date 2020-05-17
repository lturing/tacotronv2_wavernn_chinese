import tensorflow as tf

step = 1e5
model_path = './logs-Tacotron-2/taco_pretrained/tacotron_model.ckpt-{}'.format(step)
#reader = pywrap_tensorflow.NewCheckpointReader(model_path)
reader = tf.train.NewCheckpointReader(model_path)
  
## 用reader获取变量字典,key是变量名,value是变量的shape
var_to_shape_map = reader.get_variable_to_shape_map()
for var_name in var_to_shape_map.keys():
    #用reader获取变量值
    var_value = reader.get_tensor(var_name)
  
    print(var_name)
    #print("var_value",var_value)

