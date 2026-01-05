# MODEL="contriever"
# DATA="time_sensitive_qa"
# EXP_NAME="baseline_our_dataset"
# BATCH_SIZE="2048"
# LORA=""

# jid1=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/madaptor/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

# sbatch --dependency=afterok:$jid1 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/madaptor/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA

MODEL="bge"
DATA="time_sensitive_qa"
EXP_NAME="baseline_our_dataset"
BATCH_SIZE="2048"
LORA=""

jid2=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/madaptor/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid2 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/madaptor/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA


MODEL="bgem3"
DATA="time_sensitive_qa"
EXP_NAME="baseline_our_dataset"
BATCH_SIZE="2048"
LORA=""

jid3=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/madaptor/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid3 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/madaptor/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA

MODEL="gte"
DATA="time_sensitive_qa"
EXP_NAME="baseline_our_dataset"
BATCH_SIZE="2048"
LORA=""

jid4=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/madaptor/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid4 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/madaptor/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA


MODEL="gte1.5"
DATA="time_sensitive_qa"
EXP_NAME="baseline_our_dataset"
BATCH_SIZE="2048"
LORA=""

jid5=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/madaptor/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid5 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/madaptor/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA


MODEL="nomic"
DATA="time_sensitive_qa"
EXP_NAME="baseline_our_dataset"
BATCH_SIZE="2048"
LORA=""

jid6=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/madaptor/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid6 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/madaptor/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA