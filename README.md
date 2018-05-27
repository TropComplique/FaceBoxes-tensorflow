# FaceBoxes-tensorflow

This is implementation of [FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://arxiv.org/abs/1708.05234).  
Note: it will work only on GPU because I use `NCHW` format.

## How to train

For training I use `train`+`val` parts of the WIDER dataset.
It is 16106 images in total (12880 + 3226).  
For evaluation during the training I use the FDDB dataset (2845 images).
I use `AP@IOU=0.5` metrics (it is not like in the original FDDB evaluation, but like in PASCAL VOC Challenge).


1. Run `explore_and_prepare_WIDER.ipynb` to prepare the WIDER dataset   
(You will need to combine two parts using `cp train_part2/* train/ -a`).
2. Run `explore_and_prepare_FDDB.ipynb` to prepare the FDDB dataset.
3. Create tfrecords:
  ```
  python create_tfrecords.py \
      --image_dir=/home/gpu2/hdd/dan/WIDER/train/images/ \
      --annotations_dir=/home/gpu2/hdd/dan/WIDER/train/annotations/ \
      --output=data/train_shards/ \
      --num_shards=150

  python create_tfrecords.py \
      --image_dir=/home/gpu2/hdd/dan/FDDB/val/images/ \
      --annotations_dir=/home/gpu2/hdd/dan/FDDB/val/annotations/ \
      --output=data/val_shards/ \
      --num_shards=20
  ```
4. Run `python train.py` to train a face detector. Evaluation on FDDB will happen periodically.
5. Run `tensorboard --logdir=models/run00` to observe training and eval.
6. Run `python save.py` and `create_pb.py` to convert the trained model into `.pb` file.
7. Use `class` in `face_detector.py` and `.pb` file to do inference.
