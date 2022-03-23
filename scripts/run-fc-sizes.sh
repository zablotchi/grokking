#!/bin/bash
# This script should be launched from the root of the repository, i.e.
# ./scripts/run-fc-sizess.sh

fracs=($(seq 0.2 0.1 0.9))
depths=(2 8)
dims=(64 1024)
for frac in "${fracs[@]}"
  for depth in "${depths[@]}"
  do
    ./scripts/slurm-train.sh \
        model=grokk_model_fc \
        train.lr=1e-4 \
        'wandb.wandb_tags="[fc-nn, data_scaling]"' \
        dataset.frac_train=$frac \
        model.num_layers=$depth
  done

  for dim in "${dims[@]}"
  do
    ./scripts/slurm-train.sh \
        model=grokk_model_fc \
        train.lr=1e-4 \
        'wandb.wandb_tags="[fc-nn, data_scaling]"' \
        dataset.frac_train=$frac \
        model.hidden_dim=$
  done
done
