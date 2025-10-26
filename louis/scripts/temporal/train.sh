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

ulimit -n 4096

export CUDA_VISIBLE_DEVICES=0
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6
# export TQDM_DISABLE=1 # Avoid logging tqdm progress bars
# export TORCH_USE_CUDA_DSA=0 # Set to 1 only if debugging
# export CUDA_LAUNCH_BLOCKING=0 # Set to 1 only if debugging

# Useful for pytorch debugging: torch.autograd.set_detect_anomaly(True)

cd /home/thuy0050/code/tevatron

python louis/scripts/temporal/train.py \
    --model bge \
    --data temporal_nobel_prize \
    --exp_name dev3 \
    --enhanced_temporal \
    --lora \
    --eval \
    --batch_size 256 \
    --num_neg 1 \
    --epoch 5
    # --matryoshka_dim_list 768 \
    # --temporal \
    # --temporal_reconstruction \
    # --pt 1.0 \
    # --qt 1.0 \
    # --qt_recon 1.0 \
    # --pt_recon 1.0 \

# python louis/scripts/temporal/train.py \
#     --model bge \
#     --data time_sensitive_qa \
#     --exp_name dev \
#     --lora \
#     --eval \
#     --batch_size 256 \
#     --num_neg 1 \
#     --epoch 5 \
#     --matryoshka_dim_list 256 768 \
#     --temporal \
#     --temporal_reconstruction \
#     --pt 1.0 \
#     --qt 1.0 \
#     --qt_recon 1.0 \
#     --pt_recon 1.0 \
    
# # python louis/scripts/temporal/train.py \
# #     --model bge \
# #     --data temporal_nobel_prize \
# #     --exp_name baseline_recon \
# #     --lora \
# #     --enhanced_temporal \
# #     --eval \
# #     --batch_size 256 \
# #     --num_neg 1 \
# #     --epoch 5 \
# #     --pt 1.0 \
# #     --qt 1.0 \
# #     --qt_recon 1.0 \
# #     --pt_recon 1.0

# # python louis/scripts/temporal/train.py \
# #     --model bge \
# #     --data time_sensitive_qa \
# #     --exp_name baseline_recon \
# #     --lora \
# #     --enhanced_temporal \
# #     --eval \
# #     --batch_size 256 \
# #     --num_neg 1 \
# #     --epoch 5 \
# #     --pt 1.0 \
# #     --qt 1.0 \
# #     --qt_recon 1.0 \
# #     --pt_recon 1.0