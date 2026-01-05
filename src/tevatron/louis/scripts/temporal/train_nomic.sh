# MODEL="nomic"
# DATA="temporal_nobel_prize"
# EXP_NAME="matryoshka_baseline_v2"
# BATCH_SIZE=256
# QT=0.0
# PT=0.0
# DISTILLATION=0.0
# CKA_REG=0.0
# DETACH_TEMPORAL=""
# TEMPORAL=""
# FILTER_FALSE_NEGATIVES=""
# GRADIENT_CHECKPOINTING=""
# ADD_CLS=""
# TEMPORAL_DIM="64"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

# MODEL="nomic"
# DATA="temporal_nobel_prize"
# EXP_NAME="temporal_filtered0.1_128_cka0.1_distill0.1"
# BATCH_SIZE=256
# QT=0.1
# PT=0.1
# DISTILLATION=0.1
# CKA_REG=0.1
# DETACH_TEMPORAL="--detach_temporal"
# TEMPORAL="--temporal"
# FILTER_FALSE_NEGATIVES="--filter_false_negatives"
# GRADIENT_CHECKPOINTING=""
# ADD_CLS=""
# TEMPORAL_DIM="128"

# bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

# EXP_NAME="matryoshka_baseline_v2"
# BATCH_SIZE=2048

# jid1=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

# sbatch --dependency=afterok:$jid1 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA

MODEL="nomic"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered0.1_128_cka0.1_distill0.1"
BATCH_SIZE=2048
LORA="--lora"

jid2=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid2 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA
