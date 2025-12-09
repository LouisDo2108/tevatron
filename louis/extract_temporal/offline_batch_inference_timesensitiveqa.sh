#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=mg61
#SBATCH --gres=gpu:A100:1
##SBATCH --nodelist=m3u000,m3u001,m3u002,m3u003,m3u004,m3u005,m3u006,m3u007,m3u008
##SBATCH --qos=fitq
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
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=0
# export TQDM_DISABLE=1 # Avoid logging tqdm progress bars
# export TORCH_USE_CUDA_DSA=0 # Set to 1 only if debugging
# export CUDA_LAUNCH_BLOCKING=0 # Set to 1 only if debugging

# Useful for pytorch debugging: torch.autograd.set_detect_anomaly(True)

cd /home/thuy0050/code/tevatron


### Scripts for the corpus
# DATA_ROOT_DIR=/home/thuy0050/mg61_scratch2/thuy0050/data/third_work
# OUTPUT_DIR_ROOT=/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron

# python louis/extract_temporal/offline_batch_inference_refine.py \
#     --model_name_or_path bert-base-uncased \
#     --dataset_name $DATA_ROOT_DIR/tevatron/Tevatron___wikipedia-nq-corpus  \
#     --dataset_path $DATA_ROOT_DIR/temporal/nobel_prize/train/corpus_temporal.jsonl \
#     --query_max_len 512 \
#     --passage_max_len 512 \
#     --per_device_train_batch_size 64


# Scripts for the training data
DATA_ROOT_DIR=/home/thuy0050/mg61_scratch2/thuy0050/data/third_work
OUTPUT_DIR_ROOT=/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron

python /home/thuy0050/code/tevatron/louis/extract_temporal/offline_batch_inference_timesensitiveqa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name $DATA_ROOT_DIR/tevatron/Tevatron___msmarco-passage \
    --dataset_path $DATA_ROOT_DIR/temporal/time_sensitive_qa/train/train.jsonl \
    --query_max_len 512 \
    --passage_max_len 512 \
    --per_device_train_batch_size 128 \
    --dataloader_num_workers 0
