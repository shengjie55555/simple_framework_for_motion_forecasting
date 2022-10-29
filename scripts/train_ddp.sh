#!/bin/bash
torchrun --nproc_per_node=1 --master_port 29501 train_ddp.py \
  --train_dir processed/vectornet/train/ \
  --val_dir processed/vectornet/val/ \
  --train_batch_size 1 \
  --val_batch_size 1 \
  --train_epoch 50 \
  --devices 0 \
  --logger_writer \
  --num_display 10 \
  --workers 0 \
#  --resume \
#  --model_path results/vectornet_20221029_115425/weights/1.pth