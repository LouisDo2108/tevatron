MODEL="contriever"
DATA="time_sensitive_qa"
EXP_NAME="ts-retriever"
BATCH_SIZE="2048"
LORA="--lora"

# jid1=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

# sbatch --dependency=afterok:$jid1 
sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA

MODEL="contriever"
DATA="time_sensitive_qa"
EXP_NAME="ts-retriever-our_dataset"
BATCH_SIZE="2048"
LORA="--lora"

# jid2=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

# sbatch --dependency=afterok:$jid2 
sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA