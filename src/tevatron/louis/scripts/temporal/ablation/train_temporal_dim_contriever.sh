MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered0.1_32"
BATCH_SIZE=256
QT=0.1
PT=0.1
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM="32"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered0.1_64"
BATCH_SIZE=256
QT=0.1
PT=0.1
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM="64"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered0.1_128"
BATCH_SIZE=256
QT=0.1
PT=0.1
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM="128"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered0.25_32"
BATCH_SIZE=256
QT=0.25
PT=0.25
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM="32"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered0.25_64"
BATCH_SIZE=256
QT=0.25
PT=0.25
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM="64"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered0.25_128"
BATCH_SIZE=256
QT=0.25
PT=0.25
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM="128"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered0.5_32"
BATCH_SIZE=256
QT=0.5
PT=0.5
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM="32"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered0.5_64"
BATCH_SIZE=256
QT=0.5
PT=0.5
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM="64"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered0.5_128"
BATCH_SIZE=256
QT=0.5
PT=0.5
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM="128"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered1.0_32"
BATCH_SIZE=256
QT=1.0
PT=1.0
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM="32"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered1.0_64"
BATCH_SIZE=256
QT=1.0
PT=1.0
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM="64"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"

MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="temporal_filtered1.0_128"
BATCH_SIZE=256
QT=1.0
PT=1.0
DISTILLATION=0.0
CKA_REG=0.0
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
GRADIENT_CHECKPOINTING=""
ADD_CLS=""
TEMPORAL_DIM="128"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_temporal.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM"