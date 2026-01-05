MODEL="bgem3"
DATA="time_sensitive_qa"
EXP_NAME="temporal_filtered0.1_add_cls_128_cka0.1_distill0.1"
BATCH_SIZE=256
QT=0.1
PT=0.1
DISTILLATION=0.1
CKA_REG=0.1
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING="--gradient_checkpointing"
ADD_CLS="--add_cls"
TEMPORAL_DIM="128"

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

# MODEL="gte1.5"
# DATA="time_sensitive_qa"
# EXP_NAME="temporal_filtered0.1_add_cls_128_cka0.5_distill0.5"
# BATCH_SIZE=256
# QT=0.1
# PT=0.1
# DISTILLATION=0.5
# CKA_REG=0.5
# DETACH_TEMPORAL="--detach_temporal"
# TEMPORAL="--temporal"
# FILTER_FALSE_NEGATIVES="--filter_false_negatives"
# GRADIENT_CHECKPOINTING="--gradient_checkpointing"
# ADD_CLS="--add_cls"
# TEMPORAL_DIM="128"

# bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

# MODEL="bge"
# DATA="time_sensitive_qa"
# EXP_NAME="temporal_filtered0.1_add_cls_128_cka0.1_distill0.1"
# BATCH_SIZE=256
# QT=0.1
# PT=0.1
# DISTILLATION=0.1
# CKA_REG=0.1
# DETACH_TEMPORAL="--detach_temporal"
# TEMPORAL="--temporal"
# FILTER_FALSE_NEGATIVES="--filter_false_negatives"
# GRADIENT_CHECKPOINTING=""
# ADD_CLS="--add_cls"
# TEMPORAL_DIM="128"

# bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"