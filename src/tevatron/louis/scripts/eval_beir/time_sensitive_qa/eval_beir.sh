# MODEL="contriever"
# DATA="time_sensitive_qa"
# EXP_NAME="temporal_filtered_cka0.1_distill0.1"
# BATCH_SIZE="2048"
# LORA="--lora"

# jid1=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

# sbatch --dependency=afterok:$jid1 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA

MODEL="bgem3"
DATA="time_sensitive_qa"
EXP_NAME="temporal_filtered0.1_add_cls_128_cka0.1_distill0.1"
BATCH_SIZE="2048"
LORA="--lora"

jid1=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid1 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA