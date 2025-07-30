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
EXP_NAME=naive_temporal_5epoch_temp0.05_lora_bf16
OUTPUT_DIR=$OUTPUT_DIR_ROOT/$DATA_NAME/$MODEL_NAME/$BACKBONE/$EXP_NAME
MODEL_DIR=$OUTPUT_DIR_ROOT/temporal_nobel_prize/$MODEL_NAME/$BACKBONE/$EXP_NAME

mkdir -p $OUTPUT_DIR

# bash /home/thuy0050/code/tevatron/louis/nanobeir_scripts/tevatron_eval_beir.sh \
#     --dataset queries \
#     --model_name_path $OUTPUT_DIR_ROOT/temporal_nobel_prize/$MODEL_NAME/$BACKBONE/$EXP_NAME \
#     --OUTPUT_DIR $OUTPUT_DIR \
#     --normalize

# Encode passages
python -m tevatron.retriever.driver.encode \
  --per_device_eval_batch_size 512 \
  --passage_max_len 512 \
  --pooling avg \
  --bf16 \
  --normalize \
  --attn_implementation sdpa \
  --dataset_name zeta-alpha-ai/NanoNQ \
  --dataset_config corpus \
  --dataset_split train \
  --encode_output_path $OUTPUT_DIR/corpus.pkl \
  --lora_name_or_path $MODEL_DIR \
  --model_name_or_path $MODEL_DIR \
  --overwrite_output_dir

# Encode queries
python -m tevatron.retriever.driver.encode \
  --per_device_eval_batch_size 512 \
  --query_max_len 512 \
  --pooling avg \
  --bf16 \
  --normalize \
  --attn_implementation sdpa \
  --encode_is_query \
  --dataset_name zeta-alpha-ai/NanoNQ \
  --dataset_config queries \
  --dataset_split train \
  --encode_output_path $OUTPUT_DIR/query.pkl \
  --lora_name_or_path $MODEL_DIR \
  --model_name_or_path $MODEL_DIR \
  --overwrite_output_dir


# Perform retrieval
set -f && OMP_NUM_THREADS=12 python -m tevatron.retriever.driver.search \
    --query_reps $OUTPUT_DIR/query.pkl \
    --passage_reps $OUTPUT_DIR/corpus.pkl \
    --depth 1000 \
    --batch_size 512 \
    --save_text \
    --save_ranking_to $OUTPUT_DIR/rank.txt

# Convert results to TREC format
python -m tevatron.utils.format.convert_result_to_trec \
    --input $OUTPUT_DIR/rank.txt \
    --output $OUTPUT_DIR/rank.trec \
    --remove_query

# Evaluate results using pyserini
python -m pyserini.eval.trec_eval -c \
  -mrecall.100 -mndcg_cut.10 -mP.10 -mrecall.10 -mrecip_rank -mmap \
  /home/thuy0050/mg61_scratch2/thuy0050/data/third_work/tevatron/zeta-alpha-ai___nano_nq/qrels.txt \
  $OUTPUT_DIR/rank.trec
