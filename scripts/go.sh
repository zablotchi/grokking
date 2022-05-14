#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./scripts/go.sh

datasets=(mod_subtract_dataset permutation_group_dataset)
dropouts=(0 0.01 0.02 0.05 0.1 0.15 0.2)
for dataset in "${datasets[@]}"
do
  for dropout in "${dropouts[@]}"
  do
    ./scripts/slurm-train.sh \
      dataset=$dataset \
      model.transformer_config.dropout=$dropout \
      'wandb.wandb_tags="[transformer, dropout-ablation]"'
  done
done
