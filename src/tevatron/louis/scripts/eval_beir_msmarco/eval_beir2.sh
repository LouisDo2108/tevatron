cd /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir_msmarco/
DATA="temporal_nobel_prize" # time_sensitive_qa
EXP_NAME="t_128_alpha_0.1"
LORA="--lora"
# LORA=""
METHOD_NAME="tmrl" # ts-retriever, tempretriever, zero-shot

MODEL="bge"
BATCH_SIZE="2048"

jid1=$(sbatch --parsable eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")
sbatch --dependency=afterok:$jid1 eval_beir_query.sh $"$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"

# MODEL="bge"
# BATCH_SIZE="2048"

# jid2=$(sbatch --parsable eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME)
# sbatch --dependency=afterok:$jid2 eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME


# MODEL="bgem3"
# BATCH_SIZE="2048"

# jid3=$(sbatch --parsable eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME)
# sbatch --dependency=afterok:$jid3 eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME

# MODEL="gte"
# BATCH_SIZE="2048"

# jid4=$(sbatch --parsable eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME)
# sbatch --dependency=afterok:$jid4 eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME


# MODEL="gte1.5"
# BATCH_SIZE="2048"

# jid5=$(sbatch --parsable eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME)
# sbatch --dependency=afterok:$jid5 eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME


# MODEL="nomic"
# BATCH_SIZE="2048"

# jid6=$(sbatch --parsable eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME)
# sbatch --dependency=afterok:$jid6 eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME