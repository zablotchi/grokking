#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./scripts/run-fracs.sh

datasets=( mod_subtract_dataset permutation_group_dataset )
fracs=($(seq 0.1 0.1 0.9))
for dataset in "${datasets[@]}"
do
  for frac in "${fracs[@]}"
  do
    ./scripts/slurm-train.sh \
      model=grokk_model_fc \
      train.lr=1e-4 \
      'wandb.wandb_tags="[fc-nn, data_scaling]"' \
      dataset=$dataset \
      dataset.frac_train=$frac
  done
done
