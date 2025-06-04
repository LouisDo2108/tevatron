#!/bin/bash
#SBATCH --partition=fit
#SBATCH --account=ft49
#SBATCH --gres=gpu:A100:1
#SBATCH --qos=fitq
#SBATCH --job-name=thuy0050
#SBATCH --output=/home/thuy0050/code/MixLoraDSI/logs/slurm-%x-%j.out
#SBATCH --error=/home/thuy0050/code/MixLoraDSI/logs/slurm-%x-%j.err
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
OUTPUT_DIR=/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron
EXP_NAME=model_scifact_bge

# ==== TRAIN RETRIEVER ====
python src/tevatron/retriever/driver/train.py \
  --do_train \
  --fp16 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --attn_implementation sdpa \
  --dataset_name Tevatron/scifact \
  --model_name_or_path BAAI/bge-base-en-v1.5 \
  --output_dir $OUTPUT_DIR/$EXP_NAME \
  --overwrite_output_dir

# ==== ENCODE CORPUS ==== 
python src/tevatron/retriever/driver/encode.py \
  --per_device_eval_batch_size 128 \
  --passage_max_len 512 \
  --fp16 \
  --attn_implementation sdpa \
  --dataset_name Tevatron/scifact-corpus \
  --model_name_or_path $OUTPUT_DIR/$EXP_NAME \
  --encode_output_path $OUTPUT_DIR/$EXP_NAME/corpus_emb.pkl

# ==== ENCODE QUERIES ==== 
python src/tevatron/retriever/driver/encode.py \
  --per_device_eval_batch_size 128 \
  --query_max_len 64 \
  --fp16 \
  --attn_implementation sdpa \
  --encode_is_query \
  --dataset_name Tevatron/scifact \
  --dataset_split dev \
  --model_name_or_path $OUTPUT_DIR/$EXP_NAME \
  --encode_output_path $OUTPUT_DIR/$EXP_NAME/queries_emb.pkl
  