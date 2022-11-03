#!/bin/bash
python eval.py \
  --model_name atds_20221103_163709 \
  --weights_name 50 \
  --val_dir ./processed/lanegcn/train/ \
  --val_batch_size 2 \
  --model atds \
  --resume \
