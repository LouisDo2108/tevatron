DATA="time_sensitive_qa"
# DATA="temporal_nobel_prize"
BATCH_SIZE=256
DISTILLATION=0.1
CKA_REG=0.1
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
METHOD_NAME="tmrl"
ENHANCED_TEMPORAL="--enhanced_temporal"

## MODEL SPECIFIC SETTINGS ###
MODEL="contriever"
QT=0.1
PT=0.1
TEMPORAL_DIM="128"
ADD_CLS=""
GRADIENT_CHECKPOINTING=""
EXP_NAME="t_${TEMPORAL_DIM}_alpha_${QT}"

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/efficiency_benchmark.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"


# MODEL="bge"
# QT=0.1
# PT=0.1
# TEMPORAL_DIM="128"
# ADD_CLS="--add_cls"
# GRADIENT_CHECKPOINTING=""
# EXP_NAME="t_${TEMPORAL_DIM}_alpha_${QT}"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"


# MODEL="gte"
# QT=0.1
# PT=0.1
# TEMPORAL_DIM="64"
# ADD_CLS=""
# GRADIENT_CHECKPOINTING=""
# EXP_NAME="t_${TEMPORAL_DIM}_alpha_${QT}"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"


# MODEL="gte1.5"
# QT=0.25
# PT=0.25
# TEMPORAL_DIM="64"
# ADD_CLS="--add_cls"
# GRADIENT_CHECKPOINTING="--gradient_checkpointing"
# EXP_NAME="t_${TEMPORAL_DIM}_alpha_${QT}"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# # Original data
# MODEL="bgem3"
# QT=0.1
# PT=0.1
# TEMPORAL_DIM="128"
# ADD_CLS="--add_cls"
# GRADIENT_CHECKPOINTING="--gradient_checkpointing"
# EXP_NAME="t_${TEMPORAL_DIM}_alpha_${QT}"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"


# MODEL="nomic"
# QT=0.1
# PT=0.1
# TEMPORAL_DIM="128"
# ADD_CLS=""
# GRADIENT_CHECKPOINTING=""
# EXP_NAME="t_${TEMPORAL_DIM}_alpha_${QT}"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"
