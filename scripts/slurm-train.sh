#!/bin/bash
# This script should be run as ./scripts/slurm-train.sh
#
# For details on how this script works, see the SLURM guide here:
# https://supercloud.mit.edu/submitting-jobs.
#
# Also see https://stackoverflow.com/a/44168719/1337463.

sbatch <<EOT
#!/bin/bash
# Slurm sbatch options
#SBATCH -o logs/train.log-%j
#SBATCH -c 16
#SBATCH --gres=gpu:volta:1

# Load conda environment.
# See https://github.com/conda/conda/issues/7980#issuecomment-441358406
# for details on why we do it this way.
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2021b/etc/profile.d/conda.sh
conda activate grokking

# Set up and launch proxy
source scripts/rand-mallory.sh
{ mallory -config \$TMP_MALLORY_CONFIG; } &

# Utility function that waits until n background jobs are left running
# First argument is n
function waitUntilNJobsRemain() {
  local n_jobs=\$(jobs -rp | wc -l)
  echo "Waiting on \$n_jobs background jobs until \$1 jobs left..."
  while [[ \$n_jobs -gt \$1 ]]; do
    n_jobs=\$(jobs -rp | wc -l)
    echo -n "."
    sleep 1
  done
  echo ""
}

# Experiment loop
fracs=(\$(seq 0.05 0.05 0.95))
for frac in "\${fracs[@]}"
do
  # Run 4 experiments in parallel
  waitUntilNJobsRemain 4

  # Run experiment
  { python -m grokk_replica.train dataset.frac_train=\$frac $@; } &
done

# Wait until all experiments finish.
waitUntilNJobsRemain 1

# Then kill mallory
pkill -P \$\$
EOT
