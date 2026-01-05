MODEL="contriever"
DATA="time_sensitive_qa"
EXP_NAME="baseline-our-dataset"
BATCH_SIZE="2048"

jid1=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE)

sbatch --dependency=afterok:$jid1 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE

# MODEL="contriever"
# DATA="temporal_nobel_prize"
# EXP_NAME="baseline"
# BATCH_SIZE="2048"

# jid1=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE)

# sbatch --dependency=afterok:$jid1 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE

# MODEL="contriever"
# DATA="temporal_nobel_prize"
# EXP_NAME="baseline-our-dataset"
# BATCH_SIZE="2048"

# jid2=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE)

# sbatch --dependency=afterok:$jid2 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE

# MODEL="gte"
# DATA="temporal_nobel_prize"
# EXP_NAME="baseline"
# BATCH_SIZE="2048"

# jid3=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE)

# sbatch --dependency=afterok:$jid3 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE

# MODEL="gte"
# DATA="temporal_nobel_prize"
# EXP_NAME="baseline-our-dataset"
# BATCH_SIZE="2048"

# jid4=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE)

# sbatch --dependency=afterok:$jid4 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE

# MODEL="gte1.5"
# DATA="temporal_nobel_prize"
# EXP_NAME="baseline"
# BATCH_SIZE="2048"

# jid5=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE)

# sbatch --dependency=afterok:$jid5 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE

# MODEL="gte1.5"
# DATA="temporal_nobel_prize"
# EXP_NAME="baseline-our-dataset"
# BATCH_SIZE="2048"

# jid6=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE)

# sbatch --dependency=afterok:$jid6 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE

# MODEL="bge"
# DATA="temporal_nobel_prize"
# EXP_NAME="baseline"
# BATCH_SIZE="2048"

# jid7=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE)

# sbatch --dependency=afterok:$jid7 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE

# MODEL="bge"
# DATA="temporal_nobel_prize"
# EXP_NAME="baseline-our-dataset"
# BATCH_SIZE="2048"

# jid8=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE)

# sbatch --dependency=afterok:$jid8 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE

# MODEL="nomic"
# DATA="temporal_nobel_prize"
# EXP_NAME="baseline"
# BATCH_SIZE="2048"

# jid9=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE)

# sbatch --dependency=afterok:$jid9 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE

# MODEL="nomic"
# DATA="temporal_nobel_prize"
# EXP_NAME="baseline-our-dataset"
# BATCH_SIZE="2048"

# jid10=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE)

# sbatch --dependency=afterok:$jid10 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE

# MODEL="bgem3"
# DATA="temporal_nobel_prize"
# EXP_NAME="baseline"
# BATCH_SIZE="1536"

# jid11=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE)

# sbatch --dependency=afterok:$jid11 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE

# MODEL="bgem3"
# DATA="temporal_nobel_prize"
# EXP_NAME="baseline-our-dataset"
# BATCH_SIZE="1536"

# jid12=$(sbatch --parsable /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_corpus.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE)

# sbatch --dependency=afterok:$jid12 /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/eval_beir/tempretriever/eval_beir_query.sh $MODEL $DATA $EXP_NAME $BATCH_SIZE