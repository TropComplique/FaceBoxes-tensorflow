import tensorflow as tf
import json
import os

from model import model_fn
from src.input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')


CONFIG = 'config.json'
GPU_TO_USE = '0'


params = json.load(open(CONFIG))
model_params = params['model_params']
input_params = params['input_pipeline_params']


def get_input_fn(is_training=True):

    image_size = input_params['image_size'] if is_training else None
    # (for evaluation i use images of different sizes)
    dataset_path = input_params['train_dataset'] if is_training else input_params['val_dataset']
    batch_size = input_params['batch_size'] if is_training else 1
    # for evaluation it's important to set batch_size to 1

    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(
                filenames,
                batch_size=batch_size, image_size=image_size,
                repeat=is_training, shuffle=is_training,
                augmentation=is_training
            )
            features, labels = pipeline.get_batch()
        return features, labels

    return input_fn


config = tf.ConfigProto()
config.gpu_options.visible_device_list = GPU_TO_USE

run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=model_params['model_dir'],
    session_config=config,
    save_summary_steps=200,
    save_checkpoints_secs=600,
    log_step_count_steps=100
)

train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, params=model_params, config=run_config)

train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=input_params['num_steps'])
eval_spec = tf.estimator.EvalSpec(val_input_fn, steps=None, start_delay_secs=1800, throttle_secs=1800)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
