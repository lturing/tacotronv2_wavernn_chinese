import os 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf 
from tacotron.models import create_model
from tacotron_hparams import hparams
import shutil 

#with tf.device('/cpu:0'):

inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths') 

model_name = 'Tacotron'
with tf.variable_scope('Tacotron_model') as scope:
    model = create_model(model_name, hparams)
    model.initialize(inputs=inputs, input_lengths=input_lengths)


checkpoint_path = tf.train.get_checkpoint_state('./logs-Tacotron-2/taco_pretrained').model_checkpoint_path
#checkpoint_path = './logs-Tacotron-2/taco_pretrained/tacotron_model.ckpt-207000'

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, checkpoint_path)

export_path_base = './export' 
if os.path.exists(export_path_base):
    shutil.rmtree(export_path_base)

os.makedirs(export_path_base, exist_ok=True)

model_version = 1
export_path = os.path.join(
      tf.compat.as_bytes(export_path_base),
      tf.compat.as_bytes(str(model_version)))

print('Exporting trained model to', export_path)
builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)

tensor_info_inputs = tf.compat.v1.saved_model.utils.build_tensor_info(inputs)
tensor_info_input_lengths = tf.compat.v1.saved_model.utils.build_tensor_info(input_lengths)

tensor_info_mel = tf.compat.v1.saved_model.utils.build_tensor_info(model.mel_outputs[0])
tensor_info_alignment = tf.compat.v1.saved_model.utils.build_tensor_info(model.alignments[0])


prediction_signature = (
      tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs={'input': tensor_info_inputs, 'input_length': tensor_info_input_lengths},
          outputs={'mel': tensor_info_mel, 'alignment': tensor_info_alignment},
          method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME))


builder.add_meta_graph_and_variables(
    sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
    signature_def_map={
        'tacotron_fw': prediction_signature,
        #tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature,
      },
      main_op=tf.compat.v1.tables_initializer(),
      strip_default_attrs=True)

builder.save()

print('Done exporting!')

