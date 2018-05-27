import tensorflow as tf
import json
from model import model_fn


"""The purpose of this script is to export a savedmodel."""


CONFIG = 'config.json'
OUTPUT_FOLDER = 'export/run00'
GPU_TO_USE = '1'


tf.logging.set_verbosity('INFO')
params = json.load(open(CONFIG))
model_params = params['model_params']
input_pipeline_params = params['input_pipeline_params']
width, height = input_pipeline_params['image_size']

config = tf.ConfigProto()
config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=model_params['model_dir'],
    session_config=config
)
estimator = tf.estimator.Estimator(model_fn, params=model_params, config=run_config)


def serving_input_receiver_fn():
    images = tf.placeholder(dtype=tf.uint8, shape=[None, None, None, 3], name='image_tensor')
    features = {'images': tf.transpose(tf.to_float(images)*(1.0/255.0), perm=[0, 3, 1, 2])}
    return tf.estimator.export.ServingInputReceiver(features, {'images': images})


estimator.export_savedmodel(
    OUTPUT_FOLDER, serving_input_receiver_fn
)
