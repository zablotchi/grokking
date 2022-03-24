#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./scripts/run-fracs.sh

fracs=($(seq 0.1 0.05 0.95))
for frac in "${fracs[@]}"
do
  ./scripts/slurm-train.sh \
    model=grokk_model_cont_out \
    dataset=sub_dataset_cont \
    dataset.frac_train=$frac \
    'wandb.wandb_tags="[transformer, data_scaling]"'
done
