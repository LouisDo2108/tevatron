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
mamba activate tevatron

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# export TQDM_DISABLE=1 # Avoid logging tqdm progress bars
# export TORCH_USE_CUDA_DSA=0 # Set to 1 only if debugging
# export CUDA_LAUNCH_BLOCKING=0 # Set to 1 only if debugging

# Useful for pytorch debugging: torch.autograd.set_detect_anomaly(True)

cd /home/thuy0050/code/FlashRAG/scripts

python /home/thuy0050/code/FlashRAG/louis/scripts/preprocess_wiki.py \
    --dump_path /home/thuy0050/mg61_scratch2/thuy0050/data/third_work/flashrag/enwiki-20211220-pages-articles.xml.bz2 \
    --save_path /home/thuy0050/mg61_scratch2/thuy0050/data/third_work/flashrag/temp.jsonl \
    --use_chonkie \
    --chunk_by sentence \
    --seg_size 6 \
    --stride 1 \
    --num_workers 16