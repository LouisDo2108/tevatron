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
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6
# export TQDM_DISABLE=1 # Avoid logging tqdm progress bars
# export TORCH_USE_CUDA_DSA=0 # Set to 1 only if debugging
# export CUDA_LAUNCH_BLOCKING=0 # Set to 1 only if debugging

# Useful for pytorch debugging: torch.autograd.set_detect_anomaly(True)

cd /home/thuy0050/code/tevatron

DATA_ROOT_DIR=/home/thuy0050/mg61_scratch2/thuy0050/data/third_work
OUTPUT_DIR_ROOT=/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron

CHECKPOINT_DIR=facebook/contriever
DATA_NAME=timesensitiveqa
MODEL_NAME=tempretriever
BACKBONE=$CHECKPOINT_DIR
EXP_NAME=baseline
OUTPUT_DIR=$OUTPUT_DIR_ROOT/$DATA_NAME/$MODEL_NAME/$BACKBONE/$EXP_NAME

export WANDB_ENTITY=htluc19
export WANDB_PROJECT=temporal

mkdir -p $OUTPUT_DIR # Create folder if not exists

# ==== TRAIN RETRIEVER ====
python /home/thuy0050/code/tevatron/louis/madaptor/train_tempretriever.py \
  --do_train \
  --pooling cls \
  --bf16 \
  --train_group_size 5 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --temperature 0.02 \
  --logging_steps 10 \
  --num_train_epochs 10 \
  --gradient_accumulation_steps 1 \
  --attn_implementation sdpa \
  --dataset_name $DATA_ROOT_DIR/tevatron/Tevatron___msmarco-passage  \
  --dataset_path $DATA_ROOT_DIR/temporal/explitcit/time-sensitive-qa/train.jsonl \
  --model_name_or_path $CHECKPOINT_DIR \
  --run_name $BACKBONE\_$EXP_NAME \
  --output_dir $OUTPUT_DIR \
  --report_to none \
  --passage_prefix "" \
  --kl_loss


# # ==== ENCODE CORPUS ====
# python src/tevatron/retriever/driver/encode.py \
#   --per_device_eval_batch_size 512 \
#   --passage_max_len 512 \
#   --pooling cls \
#   --bf16 \
#   --normalize \
#   --attn_implementation sdpa \
#   --dataset_name LouisDo2108/temporal-nobel-prize \
#   --dataset_config corpus \
#   --encode_output_path $OUTPUT_DIR/corpus_emb.pkl \
#   --model_name_or_path $OUTPUT_DIR \
#   --lora_name_or_path $OUTPUT_DIR \
#   --overwrite_output_dir \
#   --query_prefix "" \
#   --passage_prefix ""

# # ==== ENCODE QUERIES ==== 
# python src/tevatron/retriever/driver/encode.py \
#   --per_device_eval_batch_size 512 \
#   --query_max_len 512 \
#   --pooling cls \
#   --bf16 \
#   --normalize \
#   --attn_implementation sdpa \
#   --encode_is_query \
#   --dataset_name LouisDo2108/temporal-nobel-prize \
#   --dataset_config query \
#   --model_name_or_path $OUTPUT_DIR \
#   --lora_name_or_path $OUTPUT_DIR \
#   --encode_output_path $OUTPUT_DIR/queries_emb.pkl \
#   --overwrite_output_dir \
#   --query_prefix "" \
#   --passage_prefix ""

# # ==== RETRIEVAL ====  
# set -f && OMP_NUM_THREADS=12 python -m tevatron.retriever.driver.search \
#     --query_reps $OUTPUT_DIR/queries_emb.pkl \
#     --passage_reps $OUTPUT_DIR/corpus_emb.pkl \
#     --depth 100 \
#     --batch_size 512 \
#     --save_text \
#     --save_ranking_to $OUTPUT_DIR/rank.txt

# # ==== CONVERT TO TREC FORMAT ====  
# python -m tevatron.utils.format.convert_result_to_trec \
#     --input $OUTPUT_DIR/rank.txt \
#     --output $OUTPUT_DIR/rank.trec \
#     --remove_query

# # ==== EVALUATE RESULTS USING PYSERINI ====
# # Note that the M here will set the @k (i.e., @M) of mrr and map, by default, if not set, M=100
# python -m pyserini.eval.trec_eval -c \
#   -mP.10 -mrecall.10 -mndcg_cut.10 -M 10 -mrecip_rank -mmap \
#   $DATA_ROOT_DIR/temporal/temporal_nobel_prize/test/qrel.txt \
#   $OUTPUT_DIR/rank.trec

# # # ==== CONVERT TO MSMARCO FORMAT ====  
# # python -m tevatron.utils.format.convert_result_to_marco \
# #     --input $OUTPUT_DIR/rank.txt \
# #     --output $OUTPUT_DIR/rank.msmarco \

# # # Calculate MRR@k with Pyserini's MSMARCO script
# # python -m pyserini.eval.msmarco_passage_eval \
# #   $DATA_ROOT_DIR/temporal/temporal_nobel_prize/test/qrel.txt \
# #   $OUTPUT_DIR/rank.msmarco

# python louis/beir_scripts/eval_nanobeir_with_sbert.py \
#   --model_name_or_path $OUTPUT_DIR \
#   --nanobeir_datasets NQ \
#   --pooling mean \
#   --bf16 \
#   --query_prompts "" \
#   --corpus_prompts ""