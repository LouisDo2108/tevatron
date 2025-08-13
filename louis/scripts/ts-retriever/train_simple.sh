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

DATA_NAME=ChroniclingAmericaQA
MODEL_NAME=ts-retriever
BACKBONE=bge-base-en-v1.5
EXP_NAME=baseline
OUTPUT_DIR=$OUTPUT_DIR_ROOT/$DATA_NAME/$MODEL_NAME/$BACKBONE/$EXP_NAME

CHECKPOINT_DIR=BAAI/bge-base-en-v1.5

export WANDB_ENTITY=htluc19
export WANDB_PROJECT=temporal

mkdir -p $OUTPUT_DIR # Create folder if not exists

# negative_size = self.data_args.train_group_size - 1
# lora target modules for contriever: query,key,value,dense,word_embeddings,position_embeddings
# ==== TRAIN RETRIEVER ====
# --num_train_epochs 10 \
python src/tevatron/retriever/driver/train.py \
  --do_train \
  --pooling cls \
  --bf16 \
  --normalize \
  --train_group_size 5 \
  --query_max_len 512 \
  --passage_max_len 512 \
  --per_device_train_batch_size 64 \
  --learning_rate 1e-4 \
  --temperature 0.02 \
  --logging_steps 100 \
  --attn_implementation sdpa \
  --num_train_epochs 5 \
  --lora \
  --lora_r 4 \
  --lora_alpha 16 \
  --lora_target_modules layer.11.intermediate.dense,layer.11.output.dense \
  --passage_prefix "Represent this sentence for searching relevant passages: " \
  --dataset_name $DATA_ROOT_DIR/tevatron/Tevatron___msmarco-passage  \
  --dataset_path $DATA_ROOT_DIR/temporal/temporal_nobel_prize/train/train_temporal_v2.jsonl \
  --eval_dataset_path $DATA_ROOT_DIR/temporal/temporal_nobel_prize/train/dev.jsonl \
  --model_name_or_path $CHECKPOINT_DIR \
  --run_name $BACKBONE\_$EXP_NAME \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --report_to none
  # --normalize \
  # --filter_false_negatives

# # --num_train_epochs 5 \

# # layer.11.intermediate.dense,layer.11.output.dense \
# # layer.11.intermediate.dense,layer.11.output.dense \
# layer.11.attention.self.query,layer.11.attention.self.key,layer.11.attention.self.value,

# Scaled Dot Product Attention, for BERT
# ==== ENCODE CORPUS ====
python src/tevatron/retriever/driver/encode.py \
  --per_device_eval_batch_size 512 \
  --passage_max_len 512 \
  --pooling cls \
  --bf16 \
  --normalize \
  --attn_implementation sdpa \
  --dataset_name LouisDo2108/temporal-nobel-prize \
  --dataset_config corpus \
  --encode_output_path $OUTPUT_DIR/corpus_emb.pkl \
  --model_name_or_path $OUTPUT_DIR \
  --lora_name_or_path $OUTPUT_DIR \
  --overwrite_output_dir

# ==== ENCODE QUERIES ==== 
python src/tevatron/retriever/driver/encode.py \
  --per_device_eval_batch_size 512 \
  --query_max_len 512 \
  --pooling cls \
  --bf16 \
  --normalize \
  --attn_implementation sdpa \
  --encode_is_query \
  --query_prefix "Represent this sentence for searching relevant passages: " \
  --dataset_name LouisDo2108/temporal-nobel-prize \
  --dataset_config query \
  --model_name_or_path $OUTPUT_DIR \
  --lora_name_or_path $OUTPUT_DIR \
  --encode_output_path $OUTPUT_DIR/queries_emb.pkl \
  --overwrite_output_dir

# ==== RETRIEVAL ====  
set -f && OMP_NUM_THREADS=12 python -m tevatron.retriever.driver.search \
    --query_reps $OUTPUT_DIR/queries_emb.pkl \
    --passage_reps $OUTPUT_DIR/corpus_emb.pkl \
    --depth 100 \
    --batch_size 512 \
    --save_text \
    --save_ranking_to $OUTPUT_DIR/rank.txt

# ==== CONVERT TO TREC FORMAT ====  
python -m tevatron.utils.format.convert_result_to_trec \
    --input $OUTPUT_DIR/rank.txt \
    --output $OUTPUT_DIR/rank.trec \
    --remove_query

# ==== EVALUATE RESULTS USING PYSERINI ====
# Note that the M here will set the @k (i.e., @M) of mrr and map, by default, if not set, M=100
python -m pyserini.eval.trec_eval -c \
  -mP.10 -mrecall.10 -mndcg_cut.10 -M 10 -mrecip_rank -mmap \
  $DATA_ROOT_DIR/temporal/temporal_nobel_prize/test/qrel.txt \
  $OUTPUT_DIR/rank.trec

# ==== CONVERT TO MSMARCO FORMAT ====  
python -m tevatron.utils.format.convert_result_to_marco \
    --input $OUTPUT_DIR/rank.txt \
    --output $OUTPUT_DIR/rank.msmarco \

# Calculate MRR@k with Pyserini's MSMARCO script
python -m pyserini.eval.msmarco_passage_eval \
  $DATA_ROOT_DIR/temporal/temporal_nobel_prize/test/qrel.txt \
  $OUTPUT_DIR/rank.msmarco

python louis/beir_scripts/eval_nanobeir_with_sbert.py \
  --model_name_or_path $OUTPUT_DIR \
  --nanobeir_datasets NQ \
    --query_prompts "Represent this sentence for searching relevant passages: " \
    --pooling cls \
  --bf16