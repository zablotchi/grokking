#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./scripts/run-fracs.sh

fracs=($(seq 0.1 0.05 0.6))
for frac in "${fracs[@]}"
do
  ./scripts/slurm-train.sh \
    dataset.frac_train=$frac \
    dataset=permutation_group_dataset \
    'wandb.wandb_tags="[transformer, data_scaling, S5]"'
done
