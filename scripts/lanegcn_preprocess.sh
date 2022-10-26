echo "-- Processing val set..."
python feature_extractor.py \
  --data_dir ~/code_space/python/dataset/Argoverse/val/data/ \
  --save_dir ./processed/lanegcn/ \
  --model LaneGCN \
  --mode val \
  --debug \
  --viz