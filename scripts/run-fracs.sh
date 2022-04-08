#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./scripts/run-fracs.sh

fracs=($(seq 0.01 0.01 0.1) $(seq 0.12 0.02 0.3) $(seq 0.35 0.05 0.7) 0.8 0.9)
for frac in "${fracs[@]}"
do
  ./scripts/slurm-train.sh \
    model=grokk_model_cont_out \
    dataset=sub_dataset_cont \
    dataset.frac_train=$frac \
    'wandb.wandb_tags="[transformer, data_scaling, 2_fc_layers]"'
done
