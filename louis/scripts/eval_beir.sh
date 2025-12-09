MODEL="contriever"
DATA="temporal_nobel_prize"
EXP_NAME="dev"
BATCH_SIZE="1536"
LORA="--lora"

jid1=$(sbatch --parsable /home/thuy0050/code/tevatron/louis/scripts/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid1 /home/thuy0050/code/tevatron/louis/scripts/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA

MODEL="gte"
DATA="temporal_nobel_prize"
EXP_NAME="dev"
BATCH_SIZE="1536"
LORA="--lora"

jid2=$(sbatch --parsable /home/thuy0050/code/tevatron/louis/scripts/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid2 /home/thuy0050/code/tevatron/louis/scripts/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA

MODEL="bge"
DATA="temporal_nobel_prize"
EXP_NAME="dev"
BATCH_SIZE="1536"
LORA="--lora"

jid3=$(sbatch --parsable /home/thuy0050/code/tevatron/louis/scripts/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid3 /home/thuy0050/code/tevatron/louis/scripts/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA

MODEL="nomic"
DATA="temporal_nobel_prize"
EXP_NAME="dev"
BATCH_SIZE="1280"
LORA="--lora"

jid4=$(sbatch --parsable /home/thuy0050/code/tevatron/louis/scripts/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid4 /home/thuy0050/code/tevatron/louis/scripts/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA

MODEL="gte1.5"
DATA="temporal_nobel_prize"
EXP_NAME="dev"
BATCH_SIZE="1280"
LORA="--lora"

jid5=$(sbatch --parsable /home/thuy0050/code/tevatron/louis/scripts/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid5 /home/thuy0050/code/tevatron/louis/scripts/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA

MODEL="bgem3"
DATA="temporal_nobel_prize"
EXP_NAME="dev"
BATCH_SIZE="1024"
LORA="--lora"

jid5=$(sbatch --parsable /home/thuy0050/code/tevatron/louis/scripts/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid6 /home/thuy0050/code/tevatron/louis/scripts/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA

MODEL="qwen3"
DATA="temporal_nobel_prize"
EXP_NAME="dev"
BATCH_SIZE="384"
LORA="--lora"

jid7=$(sbatch --parsable /home/thuy0050/code/tevatron/louis/scripts/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA)

sbatch --dependency=afterok:$jid7 /home/thuy0050/code/tevatron/louis/scripts/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE $LORA