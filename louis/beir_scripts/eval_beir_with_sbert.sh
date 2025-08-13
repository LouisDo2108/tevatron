#!/bin/bash
#SBATCH --partition=fit
#SBATCH --account=ft49
#SBATCH --gres=gpu:A100:1
#SBATCH --qos=fitq
#SBATCH --job-name=thuy0050
#SBATCH --output=/home/thuy0050/code/tevatron/louis/logs/slurm-%x-%j.out
#SBATCH --error=/home/thuy0050/code/tevatron/louis/logs/slurm-%x-%j.err
#SBATCH --time=1-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

#SBATCH --mail-user=tuan.huynh1@monash.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ==== ENVIRONMENT SETUP ====
source ~/.bashrc
conda activate tevatron

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# export TQDM_DISABLE=1 # Avoid logging tqdm progress bars
# export TORCH_USE_CUDA_DSA=0 # Set to 1 only if debugging
# export CUDA_LAUNCH_BLOCKING=0 # Set to 1 only if debugging

# Useful for pytorch debugging: torch.autograd.set_detect_anomaly(True)

cd /home/thuy0050/code/tevatron

DATA_ROOT_DIR=/home/thuy0050/mg61_scratch2/thuy0050/data/third_work
OUTPUT_DIR_ROOT=/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron

DATA_NAME=nanobeir/nq
MODEL_NAME=ts-retriever
BACKBONE=contriever
EXP_NAME=naive_temporal_v2_5epoch_temp0.05_lora_bf16_with_temporal_projector
OUTPUT_DIR=$OUTPUT_DIR_ROOT/$DATA_NAME/$MODEL_NAME/$BACKBONE/$EXP_NAME
MODEL_DIR=$OUTPUT_DIR_ROOT/temporal_nobel_prize/$MODEL_NAME/$BACKBONE/$EXP_NAME

python nanobeir_scripts/eval_nanobeir_with_sbert.py
