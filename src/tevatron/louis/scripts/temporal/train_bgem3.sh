# MODEL="bgem3"
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
# GRADIENT_CHECKPOINTING="--gradient_checkpointing"
# ADD_CLS="--add_cls"
# TEMPORAL_DIM="64"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

# MODEL="bgem3"
# DATA="temporal_nobel_prize"
# EXP_NAME="temporal_filtered0.25_add_cls_cka0.1_distill0.1"
# BATCH_SIZE=256
# QT=0.25
# PT=0.25
# DISTILLATION=0.1
# CKA_REG=0.1
# DETACH_TEMPORAL="--detach_temporal"
# TEMPORAL="--temporal"
# FILTER_FALSE_NEGATIVES="--filter_false_negatives"
# GRADIENT_CHECKPOINTING="--gradient_checkpointing"
# ADD_CLS="--add_cls"
# TEMPORAL_DIM="64"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

# MODEL="bgem3"
# DATA="temporal_nobel_prize"
# EXP_NAME="temporal_filtered0.1_add_cls_128_cka0.1_distill0.1_v2"
# BATCH_SIZE=256
# QT=0.1
# PT=0.1
# DISTILLATION=0.1
# CKA_REG=0.1
# DETACH_TEMPORAL="--detach_temporal"
# TEMPORAL="--temporal"
# FILTER_FALSE_NEGATIVES="--filter_false_negatives"
# GRADIENT_CHECKPOINTING="--gradient_checkpointing"
# ADD_CLS="--add_cls"
# TEMPORAL_DIM="128"

# bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"


# MODEL="bgem3"
# EXP_NAME="matryoshka_baseline_v2"
# DATA="temporal_nobel_prize"
# BATCH_SIZE=1536
# LORA="--lora"

# jid1=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

# sbatch --dependency=afterok:$jid1 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA

MODEL="bgem3"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered128_add_cls_cka0.1_distill0.1"
BATCH_SIZE=1536
LORA="--lora"

jid2=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid2 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA
