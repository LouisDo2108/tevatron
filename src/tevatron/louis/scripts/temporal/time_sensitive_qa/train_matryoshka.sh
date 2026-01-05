MODEL="bge"
DATA="temporal_nobel_prize"
EXP_NAME="matryoshka_original_dataset"
BATCH_SIZE=256
QT=0.0
PT=0.0
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL=""
TEMPORAL=""
FILTER_FALSE_NEGATIVES=""
GRADIENT_CHECKPOINTING="--gradient_checkpointing"
ADD_CLS=""

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal2.sh \
"$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" $PT "$DISTILLATION" "$CKA_REG" \
"$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS"

# MODEL="gte1.5"
# DATA="time_sensitive_qa"
# EXP_NAME="matryoshka_baseline_original_data"
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

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal2.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $QT $PT $DISTILLATION $CKA_REG $DETACH_TEMPORAL $TEMPORAL $FILTER_FALSE_NEGATIVES $GRADIENT_CHECKPOINTING $ADD_CLS


# MODEL="nomic"
# DATA="time_sensitive_qa"
# EXP_NAME="matryoshka_baseline"
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

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $QT $PT $DISTILLATION $CKA_REG $DETACH_TEMPORAL $TEMPORAL $FILTER_FALSE_NEGATIVES $GRADIENT_CHECKPOINTING $ADD_CLS

# MODEL="nomic"
# DATA="time_sensitive_qa"
# EXP_NAME="matryoshka_baseline_original_data"
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

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal2.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $QT $PT $DISTILLATION $CKA_REG $DETACH_TEMPORAL $TEMPORAL $FILTER_FALSE_NEGATIVES $GRADIENT_CHECKPOINTING $ADD_CLS


# MODEL="bgem3"
# DATA="time_sensitive_qa"
# EXP_NAME="matryoshka_baseline"
# BATCH_SIZE=256
# QT=0.0
# PT=0.0
# DISTILLATION=0.0
# CKA_REG=0.0
# DETACH_TEMPORAL=""
# TEMPORAL=""
# FILTER_FALSE_NEGATIVES=""
# GRADIENT_CHECKPOINTING="--gradient_checkpointing"
# ADD_CLS=""

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $QT $PT $DISTILLATION $CKA_REG $DETACH_TEMPORAL $TEMPORAL $FILTER_FALSE_NEGATIVES $GRADIENT_CHECKPOINTING $ADD_CLS

# MODEL="bgem3"
# DATA="time_sensitive_qa"
# EXP_NAME="matryoshka_baseline_original_data"
# BATCH_SIZE=256
# QT=0.0
# PT=0.0
# DISTILLATION=0.0
# CKA_REG=0.0
# DETACH_TEMPORAL=""
# TEMPORAL=""
# FILTER_FALSE_NEGATIVES=""
# GRADIENT_CHECKPOINTING="--gradient_checkpointing"
# ADD_CLS=""

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal2.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $QT $PT $DISTILLATION $CKA_REG $DETACH_TEMPORAL $TEMPORAL $FILTER_FALSE_NEGATIVES $GRADIENT_CHECKPOINTING $ADD_CLS
