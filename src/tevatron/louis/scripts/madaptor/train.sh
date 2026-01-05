# MODEL="contriever"
# DATA="time_sensitive_qa"
# EXP_NAME="baseline"
# BATCH_SIZE=256
# ENHANCED_TEMPORAL=""
# GRADIENT_CHECKPOINTING=""

# bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/madaptor/train_madaptor.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $ENHANCED_TEMPORAL $GRADIENT_CHECKPOINTING

# MODEL="contriever"
# DATA="time_sensitive_qa"
# EXP_NAME="baseline_our_dataset"
# BATCH_SIZE=256
# ENHANCED_TEMPORAL="--enhanced_temporal"
# GRADIENT_CHECKPOINTING=""

# bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/madaptor/train_madaptor.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $ENHANCED_TEMPORAL $GRADIENT_CHECKPOINTING

MODEL="gte"
DATA="time_sensitive_qa"
EXP_NAME="baseline"
BATCH_SIZE=256
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING=""

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/madaptor/train_madaptor.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $ENHANCED_TEMPORAL $GRADIENT_CHECKPOINTING

MODEL="gte"
DATA="time_sensitive_qa"
EXP_NAME="baseline_our_dataset"
BATCH_SIZE=256
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING=""

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/madaptor/train_madaptor.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $ENHANCED_TEMPORAL $GRADIENT_CHECKPOINTING

MODEL="nomic"
DATA="time_sensitive_qa"
EXP_NAME="baseline"
BATCH_SIZE=256
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING=""

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/madaptor/train_madaptor.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $ENHANCED_TEMPORAL $GRADIENT_CHECKPOINTING

MODEL="nomic"
DATA="time_sensitive_qa"
EXP_NAME="baseline_our_dataset"
BATCH_SIZE=256
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING=""

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/madaptor/train_madaptor.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $ENHANCED_TEMPORAL $GRADIENT_CHECKPOINTING

MODEL="bge"
DATA="time_sensitive_qa"
EXP_NAME="baseline"
BATCH_SIZE=256
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING=""

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/madaptor/train_madaptor.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $ENHANCED_TEMPORAL $GRADIENT_CHECKPOINTING

MODEL="bge"
DATA="time_sensitive_qa"
EXP_NAME="baseline_our_dataset"
BATCH_SIZE=256
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING=""

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/madaptor/train_madaptor.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $ENHANCED_TEMPORAL $GRADIENT_CHECKPOINTING

MODEL="gte1.5"
DATA="time_sensitive_qa"
EXP_NAME="baseline"
BATCH_SIZE=256
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/madaptor/train_madaptor.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $ENHANCED_TEMPORAL $GRADIENT_CHECKPOINTING

MODEL="gte1.5"
DATA="time_sensitive_qa"
EXP_NAME="baseline_our_dataset"
BATCH_SIZE=256
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/madaptor/train_madaptor.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $ENHANCED_TEMPORAL $GRADIENT_CHECKPOINTING

MODEL="bgem3"
DATA="time_sensitive_qa"
EXP_NAME="baseline"
BATCH_SIZE=256
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/madaptor/train_madaptor.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $ENHANCED_TEMPORAL $GRADIENT_CHECKPOINTING

MODEL="bgem3"
DATA="time_sensitive_qa"
EXP_NAME="baseline_our_dataset"
BATCH_SIZE=256
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/madaptor/train_madaptor.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $ENHANCED_TEMPORAL $GRADIENT_CHECKPOINTING