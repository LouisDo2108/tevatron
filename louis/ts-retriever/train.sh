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
EXP_NAME=ts-retriever

# negative_size = self.data_args.train_group_size - 1

# ==== TRAIN RETRIEVER ====
python src/tevatron/retriever/driver/train.py \
  --do_train \
  --pooling avg \
  --fp16 \
  --train_group_size 2 \
  --query_max_len 256 \
  --passage_max_len 256 \
  --per_device_train_batch_size 64 \
  --learning_rate 1e-4 \
  --temperature 1.0 \
  --logging_steps 100 \
  --num_train_epochs 5 \
  --attn_implementation sdpa \
  --dataset_name /home/thuy0050/mg61_scratch2/thuy0050/data/third_work/tevatron/Tevatron___msmarco-passage  \
  --dataset_path /home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/train/train.jsonl \
  --model_name_or_path facebook/contriever \
  --output_dir $OUTPUT_DIR/$EXP_NAME \
  --overwrite_output_dir

# --dataset_path Modify inside the code
# data_args.dataset_path = {
#     "train": "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/train/train.jsonl",
#     "dev": "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/train/dev.jsonl",
# }

# Scaled Dot Product Attention, for BERT
# ==== ENCODE CORPUS ====
python src/tevatron/retriever/driver/encode.py \
  --per_device_eval_batch_size 512 \
  --passage_max_len 256 \
  --pooling avg \
  --fp16 \
  --normalize \
  --attn_implementation sdpa \
  --dataset_name LouisDo2108/temporal-nobel-prize \
  --dataset_config corpus \
  --encode_output_path $OUTPUT_DIR/$EXP_NAME/corpus_emb.pkl \
  --model_name_or_path $OUTPUT_DIR/$EXP_NAME \
  --overwrite_output_dir

# ==== ENCODE QUERIES ==== 
python src/tevatron/retriever/driver/encode.py \
  --per_device_eval_batch_size 512 \
  --query_max_len 256 \
  --pooling avg \
  --fp16 \
  --normalize \
  --attn_implementation sdpa \
  --encode_is_query \
  --dataset_name LouisDo2108/temporal-nobel-prize \
  --dataset_config query \
  --model_name_or_path $OUTPUT_DIR/$EXP_NAME \
  --encode_output_path $OUTPUT_DIR/$EXP_NAME/queries_emb.pkl \
  --overwrite_output_dir

# ==== RETRIEVAL ====  
EMBEDDING_DIR=$OUTPUT_DIR/$EXP_NAME
set -f && OMP_NUM_THREADS=12 python -m tevatron.retriever.driver.search \
    --query_reps $EMBEDDING_DIR/queries_emb.pkl \
    --passage_reps $EMBEDDING_DIR/corpus_emb.pkl \
    --depth 100 \
    --batch_size 512 \
    --save_text \
    --save_ranking_to $EMBEDDING_DIR/rank.txt

# ==== CONVERT TO TREC FORMAT ====  
python -m tevatron.utils.format.convert_result_to_trec \
    --input $EMBEDDING_DIR/rank.txt \
    --output $EMBEDDING_DIR/rank.trec \
    --remove_query

# ==== EVALUATE RESULTS USING PYSERINI ====
python -m pyserini.eval.trec_eval -c \
  -mP.10 -mrecall.10 -mndcg_cut.10 -mrecip_rank -mmap \
  $DATA_ROOT_DIR/temporal/nobel_prize/test/qrel.txt \
  $EMBEDDING_DIR/rank.trec

# ==== CONVERT TO MSMARCO FORMAT ====  
python -m tevatron.utils.format.convert_result_to_marco \
    --input $EMBEDDING_DIR/rank.txt \
    --output $EMBEDDING_DIR/rank.msmarco \

# Calculate MRR@k with Pyserini's MSMARCO script
python -m pyserini.eval.msmarco_passage_eval \
  $DATA_ROOT_DIR/temporal/nobel_prize/test/qrel.txt \
  $EMBEDDING_DIR/rank.msmarco