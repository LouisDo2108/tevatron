# MODEL="bge"
# DATA_NAME="time_sensitive_qa"
# MODEL_NAME="ts-retriever"
# EXP_NAME="ts-retriever"
# LORA="--lora"
# MATRYOSHKA_DIM="-1"
# FLASHRAG_CONFIG="/home/thuy0050/code/FlashRAG/louis/configs/qwen8b.yaml"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag/simple_rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

MODEL="bge"
DATA_NAME="time_sensitive_qa"
MODEL_NAME="ts-retriever"
EXP_NAME="ts-retriever-our_dataset"
LORA="--lora"
MATRYOSHKA_DIM="-1"
FLASHRAG_CONFIG="/home/thuy0050/code/FlashRAG/louis/configs/qwen8b.yaml"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag/simple_rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

# MODEL="bge"
# DATA_NAME="time_sensitive_qa"
# MODEL_NAME="tempretriever"
# EXP_NAME="baseline"
# LORA="--lora"
# MATRYOSHKA_DIM="-1"
# FLASHRAG_CONFIG="/home/thuy0050/code/FlashRAG/louis/configs/qwen8b.yaml"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag/simple_rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

# MODEL="bge"
# DATA_NAME="time_sensitive_qa"
# MODEL_NAME="tempretriever"
# EXP_NAME="baseline-our-dataset"
# LORA="--lora"
# MATRYOSHKA_DIM="-1"
# FLASHRAG_CONFIG="/home/thuy0050/code/FlashRAG/louis/configs/qwen8b.yaml"

# sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag/simple_rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"