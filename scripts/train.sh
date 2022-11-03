#!/bin/bash
python train.py \
  --train_dir processed/vectornet/train/ \
  --val_dir processed/vectornet/train/ \
  --train_batch_size 1 \
  --val_batch_size 1 \
  --train_epochs 50 \
  --logger_writer \
  --num_display 10 \
  --workers 0 \
  --model vectornet \
#  --resume \
#  --model_path results/vectornet_20221029_115425/weights/1.pth

