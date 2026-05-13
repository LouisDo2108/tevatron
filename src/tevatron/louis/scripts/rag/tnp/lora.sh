cd /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag

MODEL="bge"
DATA_NAME="temporal_nobel_prize"
MODEL_NAME="lora"
LORA="--lora"
MATRYOSHKA_DIM="-1"
FLASHRAG_CONFIG="/home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag/configs/qwen8b.yaml"


### Original Data ### 
EXP_NAME="original_data"

sbatch rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

### Our Data ### 
EXP_NAME="our_data"

sbatch rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"