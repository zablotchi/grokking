#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./scripts/run-fracs.sh

dropouts=(0 0.1)
width_scales=(1 4)
fracs=($(seq 0.01 0.04 0.1) $(seq 0.12 0.08 0.3) $(seq 0.35 0.2 0.7) 0.8 0.9)
for dropout in "${dropouts[@]}"
do
  for frac in "${fracs[@]}"
  do
    for width_scale in "${width_scales[@]}"
    do
      ./scripts/slurm-train.sh \
        model=grokk_model_cont_out \
        dataset=sub_dataset_cont \
        dataset.frac_train=$frac \
        model.transformer_config.heads=$(expr 4 \* $width_scale) \
        model.transformer_config.attn_dim=32 \
        model.transformer_config.hidden_dim=$(expr 128 \* $width_scale) \
        model.transformer_config.intermediate_dim=$(expr 128 \* $width_scale) \
        model.transformer_config.dropout=$dropout \
        'wandb.wandb_tags="[cont-plateau]"'
    done
  done
done
