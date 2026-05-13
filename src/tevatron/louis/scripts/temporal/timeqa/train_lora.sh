### GENERAL SETTINGS ###
DATA="time_sensitive_qa"
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
METHOD_NAME="lora"

### MODEL SPECIFIC SETTINGS ###

# Original data
MODEL="contriever"
EXP_NAME="original_data"
GRADIENT_CHECKPOINTING=""
ENHANCED_TEMPORAL=""

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_no_mrl_eval.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

jid1=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

sbatch --dependency=afterok:$jid1 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"

# Our data
MODEL="contriever"
EXP_NAME="our_data"
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING=""

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_no_mrl_eval.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# jid2=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

# sbatch --dependency=afterok:$jid2 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"

# Original data
MODEL="bge"
EXP_NAME="original_data"
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING=""

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_no_mrl_eval.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

jid3=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

sbatch --dependency=afterok:$jid3 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"


# Our data
MODEL="bge"
EXP_NAME="our_data"
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING=""

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_no_mrl_eval.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# jid4=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

# sbatch --dependency=afterok:$jid4 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"

# Original data
MODEL="gte"
EXP_NAME="original_data"
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING=""

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_no_mrl_eval.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

jid5=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

sbatch --dependency=afterok:$jid5 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"

# Our data
MODEL="gte"
EXP_NAME="our_data"
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING=""

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_no_mrl_eval.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# jid6=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

# sbatch --dependency=afterok:$jid6 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"

# Original data
MODEL="gte1.5"
EXP_NAME="original_data"
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_no_mrl_eval.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

jid7=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

sbatch --dependency=afterok:$jid7 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"

# Our data
MODEL="gte1.5"
EXP_NAME="our_data"
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING=""

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_no_mrl_eval.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# jid8=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

# sbatch --dependency=afterok:$jid8 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"

# Original data
MODEL="bgem3"
EXP_NAME="original_data"
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_no_mrl_eval.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

jid9=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

sbatch --dependency=afterok:$jid9 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"

# Our data
MODEL="bgem3"
EXP_NAME="our_data"
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_no_mrl_eval.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# jid10=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

# sbatch --dependency=afterok:$jid10 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"

# Original data
MODEL="nomic"
EXP_NAME="original_data"
ENHANCED_TEMPORAL=""
GRADIENT_CHECKPOINTING=""

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_no_mrl_eval.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

# jid11=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

# sbatch --dependency=afterok:$jid11 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"

# Our data
MODEL="nomic"
EXP_NAME="our_data"
ENHANCED_TEMPORAL="--enhanced_temporal"
GRADIENT_CHECKPOINTING=""

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/temporal/train_no_mrl_eval.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

jid12=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

sbatch --dependency=afterok:$jid12 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"