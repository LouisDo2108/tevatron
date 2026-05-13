cd /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag

MODEL="contriever"
DATA_NAME="time_sensitive_qa"
MODEL_NAME="tmrl"
LORA="--lora"
FLASHRAG_CONFIG="/home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag/configs/qwen8b.yaml"

EXP_NAME="t_128_alpha_0.1"

MATRYOSHKA_DIM="64"
sbatch rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

MATRYOSHKA_DIM="128"
sbatch rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

MATRYOSHKA_DIM="256"
sbatch rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

MATRYOSHKA_DIM="512"
sbatch rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

MATRYOSHKA_DIM="768"
sbatch rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

# MATRYOSHKA_DIM="1024"
# sbatch rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"