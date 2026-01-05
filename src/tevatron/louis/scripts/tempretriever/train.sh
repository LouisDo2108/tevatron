# CHECKPOINT_DIR="BAAI/bge-base-en-v1.5"
# POOLING="cls"

# bash /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/tempretriever/train_tempretriever.sh $CHECKPOINT_DIR $POOLING


CHECKPOINT_DIR="BAAI/bge-m3"
POOLING="cls"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/tempretriever/train_tempretriever.sh $CHECKPOINT_DIR $POOLING


CHECKPOINT_DIR="thenlper/gte-base"
POOLING="mean"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/tempretriever/train_tempretriever.sh $CHECKPOINT_DIR $POOLING

CHECKPOINT_DIR="Alibaba-NLP/gte-base-en-v1.5"
POOLING="cls"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/tempretriever/train_tempretriever.sh $CHECKPOINT_DIR $POOLING

CHECKPOINT_DIR="nomic-ai/nomic-embed-text-v1.5"
POOLING="mean"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/tempretriever/train_tempretriever.sh $CHECKPOINT_DIR $POOLING