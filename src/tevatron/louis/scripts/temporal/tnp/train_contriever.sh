MODEL="bge"
DATA="temporal_nobel_prize"
EXP_NAME="our_data"
BATCH_SIZE=256
QT=0.1
PT=0.1
DISTILLATION=0.1
CKA_REG=0.1
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM="64"
METHOD_NAME="tempretriever"
ENHANCED_TEMPORAL="--enhanced_temporal"

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# jid1=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL")

# jid2=$(sbatch --parsable --dependency=afterok:$jid1 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

# sbatch --dependency=afterok:$jid2 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"