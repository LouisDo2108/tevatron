cd /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag

DATA_NAME="time_sensitive_qa"
MODEL_NAME="zero-shot"
EXP_NAME="baseline"
LORA="--lora"
MATRYOSHKA_DIM="-1"
FLASHRAG_CONFIG="/home/thuy0050/code/FlashRAG/louis/configs/qwen8b.yaml"

MODEL="contriever"
bash rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

# MODEL="bge"
# bash rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"