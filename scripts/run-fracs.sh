#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./scripts/run-fracs.sh

fracs=($(seq 0.2 0.1 0.9))
ps=($(256 512))
for p in "${ps[@]}"
do
  for frac in "${fracs[@]}"
  do
    ./scripts/slurm-train.sh \
      dataset.frac_train=$frac \
      dataset.p=$p \
      'wandb.wandb_tags="[transformer, data_scaling]"'
  done
done


#      dataset=permutation_group_dataset \
