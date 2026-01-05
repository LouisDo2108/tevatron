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

# ========= Temporal Nobel Prize ==========
# DATA_NAME=temporal_nobel_prize
# OUTPUT_DIR=/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/$DATA_NAME/bm25
# mkdir -p $OUTPUT_DIR

# python /home/thuy0050/code/tevatron/louis/scripts/baselines/bm25_temporal_nobel_prize.py


# mamba activate bm25s
# # ==== CONVERT TO TREC FORMAT ====  
# python -m tevatron.utils.format.convert_result_to_trec \
#     --input $OUTPUT_DIR/rank.txt \
#     --output $OUTPUT_DIR/rank.trec \
#     --remove_query

# # ==== EVALUATE RESULTS USING PYSERINI ====
# # Note that the M here will set the @k (i.e., @M) of mrr and map, by default, if not set, M=100
# python -m pyserini.eval.trec_eval -c \
#   -m recall.10,100 -m ndcg_cut.10 -M 100 \
#   $DATA_ROOT_DIR/temporal/temporal_nobel_prize/test/qrel.txt \
#   $OUTPUT_DIR/rank.trec > $OUTPUT_DIR/bm25_out.txt

# # ========= Time sensitve qa ==========
DATA_NAME=time_sensitive_qa
OUTPUT_DIR=/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/$DATA_NAME/bm25
mkdir -p $OUTPUT_DIR

# mamba activate bm25s
# unset LD_LIBRARY_PATH
# python /home/thuy0050/code/tevatron/louis/scripts/baselines/bm25_timesensitiveqa.py

mamba deactivate
mamba activate tevatron
# ==== CONVERT TO TREC FORMAT ====  
python -m tevatron.utils.format.convert_result_to_trec \
    --input $OUTPUT_DIR/rank.txt \
    --output $OUTPUT_DIR/rank.trec \
    --remove_query

# ==== EVALUATE RESULTS USING PYSERINI ====
# Note that the M here will set the @k (i.e., @M) of mrr and map, by default, if not set, M=100
python -m pyserini.eval.trec_eval -c \
  -m recall.10,100 -m ndcg_cut.10 -M 100 \
  $DATA_ROOT_DIR/temporal/time_sensitive_qa/test/qrel.txt \
  $OUTPUT_DIR/rank.trec > $OUTPUT_DIR/out.txt

# # ========= NanoBEIR NQ==========
# DATA_NAME=nanobeir/nq
# OUTPUT_DIR=/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/$DATA_NAME/bm25
# mkdir -p $OUTPUT_DIR

# # python /home/thuy0050/code/tevatron/louis/scripts/baselines/bm25_nanobeir_nq.py

# # ==== CONVERT TO TREC FORMAT ====  
# python -m tevatron.utils.format.convert_result_to_trec \
#     --input $OUTPUT_DIR/rank.txt \
#     --output $OUTPUT_DIR/rank.trec \
#     --remove_query

# # ==== EVALUATE RESULTS USING PYSERINI ====
# # Note that the M here will set the @k (i.e., @M) of mrr and map, by default, if not set, M=100
# python -m pyserini.eval.trec_eval -c \
#   -mP.10 -mrecall.10 -mndcg_cut.10 -M 10 -mrecip_rank -mmap \
#   $DATA_ROOT_DIR/temporal/nanobeir_nq/qrel.txt \
#   $OUTPUT_DIR/rank.trec
