#!/usr/bin/env bash
set -euo pipefail

# Script to evaluate all experiments matching stjohns* on the provided dataset
# - Creates a temporary symlink test/ -> train/ under dataset if test/ is missing
# - Finds the best checkpoint under each experiment's checkpoints/ (prefers checkpoint_best_loss.pth.tar)
# - Runs train_and_evaluate.py in --eval-only mode and saves results to <experiment>/eval_on_320x180

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==== EDIT THESE ====
DATASET="/home/arl/Data/NVSPrior-w/data/image_compression/320x180"
PATTERN="stjohns*"
EVAL_BATCH_SIZE=1
EVAL_NUM_WORKERS=4
GPU_ID=0
USE_CUDA="True"   # set to "False" to force CPU
# ====================

# safety: absolute paths
DATASET_ABS="$DATASET"

echo "Script dir: $SCRIPT_DIR"
echo "Dataset: $DATASET_ABS"

# helper to clean up test symlink if we created it
cleanup_symlink=false
cleanup_path=""
function finish {
  if [ "$cleanup_symlink" = true ] && [ -L "$cleanup_path" ]; then
    echo "Removing temporary symlink: $cleanup_path"
    rm -f "$cleanup_path"
  fi
}
trap finish EXIT

# if dataset/test doesn't exist but dataset/train exists, create a symlink test->train
if [ ! -d "$DATASET_ABS/test" ]; then
  if [ -d "$DATASET_ABS/train" ]; then
    echo "No test/ found under dataset; creating symlink test -> train (temporary)"
    ln -s "$DATASET_ABS/train" "$DATASET_ABS/test"
    cleanup_symlink=true
    cleanup_path="$DATASET_ABS/test"
  else
    echo "ERROR: dataset has neither test/ nor train/ under $DATASET_ABS. Please prepare a test/ folder with images." >&2
    exit 1
  fi
fi

# Loop experiments inside playground/experiments
EXP_ROOT="$SCRIPT_DIR/experiments"
shopt -s nullglob
for expdir in "$EXP_ROOT"/${PATTERN}; do
  [ -d "$expdir" ] || continue
  expname=$(basename "$expdir")
  echo "\n==> Processing experiment: $expname"

  # try to infer lambda from folder name pattern _lambda_<lstr>
  if [[ "$expname" =~ _lambda_(.*)$ ]]; then
    lstr="${BASH_REMATCH[1]}"
    lval=$(echo "$lstr" | sed 's/p/./g; s/m/-/g')
  else
    lval="0.0018"
  fi

  # find checkpoint: prefer checkpoint_best_loss.pth.tar
  ckpt_best="$expdir/checkpoints/checkpoint_best_loss.pth.tar"
  if [ -f "$ckpt_best" ]; then
    ckpt="$ckpt_best"
  else
    # fallback to last checkpoint_*.pth.tar
    ckpt_file=$(ls -1 "$expdir"/checkpoints/checkpoint_*.pth.tar 2>/dev/null | tail -n 1 || true)
    if [ -n "$ckpt_file" ] && [ -f "$ckpt_file" ]; then
      ckpt="$ckpt_file"
    else
      echo "  No checkpoint found in $expdir/checkpoints -> skipping"
      continue
    fi
  fi

  eval_save_dir="$expdir/eval_on_320x180"
  mkdir -p "$eval_save_dir"

  echo "  Using checkpoint: $ckpt"
  echo "  Saving eval outputs to: $eval_save_dir"

  python train_and_evaluate.py \
    --eval-only \
    --train_dataset "$DATASET_ABS" \
    --eval-dataset "$DATASET_ABS" \
    --eval-checkpoint "$ckpt" \
    --eval-save-dir "$eval_save_dir" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --eval-num-workers "$EVAL_NUM_WORKERS" \
    --gpu_id "$GPU_ID" \
    --cuda "$USE_CUDA" \
    --experiment "${expname%_lambda_*}" \
    --lambda "$lval"

  echo "  Done: $expname"
done

echo "\nAll done."
