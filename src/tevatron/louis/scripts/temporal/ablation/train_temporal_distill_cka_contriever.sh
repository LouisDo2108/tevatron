# MODEL="contriever"
# DATA="temporal_nobel_prize"
# EXP_NAME="temporal_filtered0.25_add_cls_64_cka0.1"
# BATCH_SIZE=256
# QT=0.1
# PT=0.1
# DISTILLATION=0.0
# CKA_REG=0.1
# DETACH_TEMPORAL="--detach_temporal"
# TEMPORAL="--temporal"
# FILTER_FALSE_NEGATIVES="--filter_false_negatives"
# GRADIENT_CHECKPOINTING=""
# ADD_CLS=""
# TEMPORAL_DIM="64"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

# MODEL="contriever"
# DATA="temporal_nobel_prize"
# EXP_NAME="temporal_filtered0.25_add_cls_64_distill0.1"
# BATCH_SIZE=256
# QT=0.1
# PT=0.1
# DISTILLATION=0.1
# CKA_REG=0.0
# DETACH_TEMPORAL="--detach_temporal"
# TEMPORAL="--temporal"
# FILTER_FALSE_NEGATIVES="--filter_false_negatives"
# GRADIENT_CHECKPOINTING=""
# ADD_CLS=""
# TEMPORAL_DIM="64"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

# MODEL="contriever"
# DATA="temporal_nobel_prize"
# EXP_NAME="temporal_filtered0.25_add_cls_64_cka0.1_distill0.1"
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
# TEMPORAL_DIM="64"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

# MODEL="contriever"
# DATA="temporal_nobel_prize"
# EXP_NAME="temporal_filtered0.25_add_cls_64_cka0.25_distill0.25"
# BATCH_SIZE=256
# QT=0.1
# PT=0.1
# DISTILLATION=0.25
# CKA_REG=0.25
# DETACH_TEMPORAL="--detach_temporal"
# TEMPORAL="--temporal"
# FILTER_FALSE_NEGATIVES="--filter_false_negatives"
# GRADIENT_CHECKPOINTING=""
# ADD_CLS=""
# TEMPORAL_DIM="64"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

# MODEL="contriever"
# DATA="temporal_nobel_prize"
# EXP_NAME="temporal_filtered0.25_add_cls_64_cka0.5_distill0.5"
# BATCH_SIZE=256
# QT=0.1
# PT=0.1
# DISTILLATION=0.5
# CKA_REG=0.5
# DETACH_TEMPORAL="--detach_temporal"
# TEMPORAL="--temporal"
# FILTER_FALSE_NEGATIVES="--filter_false_negatives"
# GRADIENT_CHECKPOINTING=""
# ADD_CLS=""
# TEMPORAL_DIM="64"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered0.1_cka0.1"
BATCH_SIZE="2560"
LORA="--lora"

jid1=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid1 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA


EXP_NAME="temporal_filtered0.1_distill0.1"

jid2=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid2 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA


# EXP_NAME="temporal_filtered0.25_add_cls_64_cka0.1_distill0.1"

# jid3=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

# sbatch --dependency=afterok:$jid3 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA


# EXP_NAME="temporal_filtered0.25_add_cls_64_cka0.25_distill0.25"

# jid4=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

# sbatch --dependency=afterok:$jid4 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA


# EXP_NAME="temporal_filtered0.25_add_cls_64_cka0.5_distill0.5"

# jid5=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

# sbatch --dependency=afterok:$jid5 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA