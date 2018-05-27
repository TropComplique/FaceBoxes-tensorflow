import io
import os
import PIL.Image
import tensorflow as tf
import json
import shutil
import random
import math
import argparse
from tqdm import tqdm
import sys


"""
The purpose of this script is to create a set of .tfrecords files
from a folder of images and a folder of annotations.
Annotations are in the json format.
Images must have .jpg or .jpeg filename extension.

Example of a json annotation (with filename "132416.json"):
{
  "object": [
    {"bndbox": {"ymin": 20, "ymax": 276, "xmax": 1219, "xmin": 1131}, "name": "face"},
    {"bndbox": {"ymin": 1, "ymax": 248, "xmax": 1149, "xmin": 1014}, "name": "face"}
  ],
  "filename": "132416.jpg",
  "size": {"depth": 3, "width": 1920, "height": 1080}
}

Example of use:
python create_tfrecords.py \
    --image_dir=/home/gpu2/hdd/dan/WIDER/val/images/ \
    --annotations_dir=/home/gpu2/hdd/dan/WIDER/val/annotations/ \
    --output=data/train_shards/ \
    --num_shards=100
"""


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', type=str)
    parser.add_argument('-a', '--annotations_dir', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-s', '--num_shards', type=int, default=1)
    return parser.parse_args()


def dict_to_tf_example(annotation, image_dir):
    """Convert dict to tf.Example proto.

    Notice that this function normalizes the bounding
    box coordinates provided by the raw data.

    Arguments:
        data: a dict.
        image_dir: a string, path to the image directory.
    Returns:
        an instance of tf.Example.
    """
    image_name = annotation['filename']
    assert image_name.endswith('.jpg') or image_name.endswith('.jpeg')

    image_path = os.path.join(image_dir, image_name)
    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()

    # check image format
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG!')

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])
    assert width > 0 and height > 0
    assert image.size[0] == width and image.size[1] == height
    ymin, xmin, ymax, xmax = [], [], [], []

    just_name = image_name[:-4] if image_name.endswith('.jpg') else image_name[:-5]
    annotation_name = just_name + '.json'
    if len(annotation['object']) == 0:
        print(annotation_name, 'is without any objects!')

    for obj in annotation['object']:
        a = float(obj['bndbox']['ymin'])/height
        b = float(obj['bndbox']['xmin'])/width
        c = float(obj['bndbox']['ymax'])/height
        d = float(obj['bndbox']['xmax'])/width
        assert (a < c) and (b < d)
        ymin.append(a)
        xmin.append(b)
        ymax.append(c)
        xmax.append(d)
        assert obj['name'] == 'face'

    example = tf.train.Example(features=tf.train.Features(feature={
        'filename': _bytes_feature(image_name.encode()),
        'image': _bytes_feature(encoded_jpg),
        'xmin': _float_list_feature(xmin),
        'xmax': _float_list_feature(xmax),
        'ymin': _float_list_feature(ymin),
        'ymax': _float_list_feature(ymax),
    }))
    return example


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def main():
    ARGS = make_args()

    image_dir = ARGS.image_dir
    annotations_dir = ARGS.annotations_dir
    print('Reading images from:', image_dir)
    print('Reading annotations from:', annotations_dir, '\n')

    examples_list = os.listdir(annotations_dir)
    num_examples = len(examples_list)
    print('Number of images:', num_examples)

    num_shards = ARGS.num_shards
    shard_size = math.ceil(num_examples/num_shards)
    print('Number of images per shard:', shard_size)

    output_dir = ARGS.output
    shutil.rmtree(output_dir, ignore_errors=True)
    os.mkdir(output_dir)

    shard_id = 0
    num_examples_written = 0
    for example in tqdm(examples_list):

        if num_examples_written == 0:
            shard_path = os.path.join(output_dir, 'shard-%04d.tfrecords' % shard_id)
            writer = tf.python_io.TFRecordWriter(shard_path)

        path = os.path.join(annotations_dir, example)
        annotation = json.load(open(path))
        tf_example = dict_to_tf_example(annotation, image_dir)
        writer.write(tf_example.SerializeToString())
        num_examples_written += 1

        if num_examples_written == shard_size:
            shard_id += 1
            num_examples_written = 0
            writer.close()

    if num_examples_written != shard_size and num_examples % num_shards != 0:
        writer.close()

    print('Result is here:', ARGS.output)


main()
