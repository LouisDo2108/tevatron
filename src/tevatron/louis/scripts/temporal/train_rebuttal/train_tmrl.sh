BATCH_SIZE=256
DISTILLATION=0.1
CKA_REG=0.1
DETACH_TEMPORAL="--detach_temporal"
TEMPORAL="--temporal"
FILTER_FALSE_NEGATIVES="--filter_false_negatives"
METHOD_NAME="tmrl"
ENHANCED_TEMPORAL="--enhanced_temporal"
MODEL="contriever"
QT=0.1
PT=0.1
TEMPORAL_DIM="128"
ADD_CLS=""
GRADIENT_CHECKPOINTING=""

DATA="non_filtered"
EXP_NAME=$DATA

bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_rebuttal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# DATA="no_meet-met"
# EXP_NAME=$DATA

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_rebuttal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# DATA="no_start"
# EXP_NAME=$DATA

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_rebuttal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# DATA="no_finish"
# EXP_NAME=$DATA

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_rebuttal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# DATA="no_overlap"
# EXP_NAME=$DATA

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_rebuttal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# DATA="no_contain-during"
# EXP_NAME=$DATA

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_rebuttal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# DATA="no_before-after"
# EXP_NAME=$DATA

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_rebuttal/train.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# cd /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir
# DATA="time_sensitive_qa" # time_sensitive_qa
# LORA="--lora"
# METHOD_NAME="tmrl" # ts-retriever, tempretriever, zero-shot
# MODEL="contriever"
# BATCH_SIZE="2048"
# DATA="time_sensitive_qa"

# EXP_NAME="no_temporal_answer"
# # jid1=$(sbatch --parsable eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME)
# # sbatch --dependency=afterok:$jid1 
# bash eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME

# EXP_NAME="no_equal"
# # jid1=$(sbatch --parsable eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME)
# # sbatch --dependency=afterok:$jid1 
# bash eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME

# EXP_NAME="no_finish"
# # jid1=$(sbatch --parsable eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME)
# # sbatch --dependency=afterok:$jid1 
# bash eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME

# EXP_NAME="no_meet-met"
# # jid1=$(sbatch --parsable eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME)
# # sbatch --dependency=afterok:$jid1 
# bash eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME

# EXP_NAME="no_before-after"
# # jid1=$(sbatch --parsable eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME)
# # sbatch --dependency=afterok:$jid1 
# bash eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME

# EXP_NAME="no_contain-during"
# # jid1=$(sbatch --parsable eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME)
# # sbatch --dependency=afterok:$jid1 
# bash eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME

# EXP_NAME="no_overlap"
# # jid1=$(sbatch --parsable eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME)
# # sbatch --dependency=afterok:$jid1 
# bash eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME

# EXP_NAME="no_start"
# # jid1=$(sbatch --parsable eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME)
# # sbatch --dependency=afterok:$jid1 
# bash eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA $METHOD_NAME
