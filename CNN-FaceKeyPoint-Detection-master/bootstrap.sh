#!/usr/bin/env bash

python3 prototxt/generate.py

# level-1
python3 dataset/data_level1.py
rm -rf log/train1.log

echo "Training LEVEL-1 ..."
python3 train.py 1

# level-2
python3 dataset/data_level2.py
rm -rf log/train2.log

echo "Training LEVEL-2 ..."
python3 train.py 2

# level-3
python3 dataset/data_level3.py
rm -rf log/train3.log

echo "Training LEVEL-3 ..."
python3 train.py 3

echo "Training Well Done!"
