cd /home/thuy0050/code/TMRL/src/tevatron/louis/scripts/rag

MODEL="contriever"
DATA_NAME="temporal_nobel_prize"
MODEL_NAME="ts-retriever"
LORA="--lora"
MATRYOSHKA_DIM="-1"
FLASHRAG_CONFIG="/home/thuy0050/code/TMRL/src/tevatron/louis/scripts/rag/configs/qwen8b.yaml"
# FLASHRAG_CONFIG="/home/thuy0050/code/TMRL/src/tevatron/louis/scripts/rag/configs/deepseek_r1_qwen32b.yaml"
# FLASHRAG_CONFIG="/home/thuy0050/code/TMRL/src/tevatron/louis/scripts/rag/configs/llama3.3_70b.yaml"
# "/home/thuy0050/code/TMRL/src/tevatron/louis/scripts/rag/configs/qwen32b.yaml"

### Original Data ### 
EXP_NAME="bs256_numneg4/original_data_v2" # "t_64_alpha_0.1""original_data"

bash rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

### Our Data ### 
EXP_NAME="bs256_numneg4/our_data_v2" # "t_64_alpha_0.1" # EXP_NAME="our_data"

bash rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"