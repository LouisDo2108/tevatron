#!/bin/bash

## FOR PARTITION GPU, A100
##SBATCH --partition=gpu
##SBATCH --gres=gpu:A100:1
##SBATCH --nodelist=m3n100,m3n101,m3n102,m3n103,m3n104,m3n105,m3n106,m3n107,m3n108,m3n109,m3n110,m3n111,m3n112

## FOR PARTITION GPU, L40S
##SBATCH --partition=gpu
##SBATCH --gres=gpu:L40S:1

## FOR FIT PARTITION
#SBATCH --partition=fit
#SBATCH --nodelist=m3u000,m3u001,m3u002,m3u003,m3u004,m3u005,m3u006,m3u007,m3u008
#SBATCH --gres=gpu:A100:1
#SBATCH --qos=fitq

#SBATCH --account=mg61
#SBATCH --job-name=thuy0050
#SBATCH --output=/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/logs/slurm-%x-%j.out
#SBATCH --error=/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/logs/slurm-%x-%j.err
#SBATCH --time=1-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

##SBATCH --mail-user=tuan.huynh1@monash.edu
##SBATCH --mail-type=BEGIN,END,FAIL

# ==== ENVIRONMENT SETUP ====
source ~/.bashrc
mamba activate tevatron

# ulimit -n 4096

export CUDA_VISIBLE_DEVICES=0
# export PYTORCH_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6
# export TQDM_DISABLE=1 # Avoid logging tqdm progress bars
# export TORCH_USE_CUDA_DSA=0 # Set to 1 only if debugging
# export CUDA_LAUNCH_BLOCKING=0 # Set to 1 only if debugging

# Useful for pytorch debugging: torch.autograd.set_detect_anomaly(True)

cd /home/thuy0050/code/tevatron/src/tevatron/

MODEL=$1
DATA=$2
EXP_NAME=$3
BATCH_SIZE=$4
LORA=$5

# echo $MODEL $DATA $EXP_NAME $BATCH_SIZE
# echo "lora $LORA"
python louis/scripts/eval_beir/eval_beir_query.py \
  --model $MODEL \
  --data $DATA \
  --exp_name $EXP_NAME \
  --batch_size $BATCH_SIZE \
  $LORA