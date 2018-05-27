# FaceBoxes-tensorflow


1. Run `explore_and_prepare_WIDER.ipynb` to convert the datasets
2. Copy cp train_part2/* train/ -a

12880 + 3226 = 16106
2845

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