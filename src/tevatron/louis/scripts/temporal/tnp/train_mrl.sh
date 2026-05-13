### GENERAL SETTINGS ###
DATA="temporal_nobel_prize"
BATCH_SIZE=256
QT=0.0
PT=0.0
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL=""
TEMPORAL=""
FILTER_FALSE_NEGATIVES=""
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM=""
METHOD_NAME="mrl"

### MODEL SPECIFIC SETTINGS ###

# # Original data
# MODEL="contriever"
# EXP_NAME="mrl_original_data_v2"
# ENHANCED_TEMPORAL=""
# GRADIENT_CHECKPOINTING="--gradient_checkpointing"

# bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# # Our data
# MODEL="contriever"
# EXP_NAME="mrl_our_data_v2"
# ENHANCED_TEMPORAL="--enhanced_temporal"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# Original data
MODEL="bge"
EXP_NAME="mrl_original_data"
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# Our data
MODEL="bge"
EXP_NAME="mrl_our_data"
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING=""

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# Original data
MODEL="gte"
EXP_NAME="mrl_original_data"
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# Our data
MODEL="gte"
EXP_NAME="mrl_our_data"
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING=""

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# Original data
MODEL="gte1.5"
EXP_NAME="mrl_original_data"
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# Our data
MODEL="gte1.5"
EXP_NAME="mrl_our_data"
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# Original data
MODEL="bgem3"
EXP_NAME="mrl_original_data"
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# Our data
MODEL="bgem3"
EXP_NAME="mrl_our_data"
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# Original data
MODEL="nomic"
EXP_NAME="mrl_original_data"
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# Our data
MODEL="nomic"
EXP_NAME="mrl_our_data"
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING=""

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"