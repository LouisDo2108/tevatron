### GENERAL SETTINGS ###
DATA="temporal_nobel_prize"
BATCH_SIZE=64
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
METHOD_NAME="tsm"

### MODEL SPECIFIC SETTINGS ###

# Original data
MODEL="contriever"
EXP_NAME="merged" # "single_year_in" "range_between_and" "range_from_to"
ENHANCED_TEMPORAL=""

# bash /home/thuy0050/code/TMRL/src/tevatron/louis/scripts/temporal/train_no_mrl_eval.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$QT" "$PT" "$DISTILLATION" "$CKA_REG" "$DETACH_TEMPORAL" "$TEMPORAL" "$FILTER_FALSE_NEGATIVES" "$GRADIENT_CHECKPOINTING" "$ADD_CLS" "$TEMPORAL_DIM" "$METHOD_NAME" "$ENHANCED_TEMPORAL"

jid1=$(sbatch --parsable /home/thuy0050/code/TMRL/src/tevatron/louis/scripts/eval_beir/eval_beir_corpus.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME")

sbatch --dependency=afterok:$jid1 /home/thuy0050/code/TMRL/src/tevatron/louis/scripts/eval_beir/eval_beir_query.sh "$MODEL" "$DATA" "$EXP_NAME" "$BATCH_SIZE" "$LORA" "$METHOD_NAME"
