MODEL="bge"
DATA_NAME="time_sensitive_qa"
MODEL_NAME="madaptor"
EXP_NAME="baseline"
LORA="--lora"
MATRYOSHKA_DIM="64"
FLASHRAG_CONFIG="/home/thuy0050/code/FlashRAG/louis/configs/qwen8b.yaml"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag/simple_rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

MODEL="bge"
DATA_NAME="time_sensitive_qa"
MODEL_NAME="madaptor"
EXP_NAME="baseline"
LORA="--lora"
MATRYOSHKA_DIM="128"
FLASHRAG_CONFIG="/home/thuy0050/code/FlashRAG/louis/configs/qwen8b.yaml"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag/simple_rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

MODEL="bge"
DATA_NAME="time_sensitive_qa"
MODEL_NAME="madaptor"
EXP_NAME="baseline"
LORA="--lora"
MATRYOSHKA_DIM="256"
FLASHRAG_CONFIG="/home/thuy0050/code/FlashRAG/louis/configs/qwen8b.yaml"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag/simple_rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

MODEL="bge"
DATA_NAME="time_sensitive_qa"
MODEL_NAME="madaptor"
EXP_NAME="baseline"
LORA="--lora"
MATRYOSHKA_DIM="512"
FLASHRAG_CONFIG="/home/thuy0050/code/FlashRAG/louis/configs/qwen8b.yaml"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag/simple_rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"

MODEL="bge"
DATA_NAME="time_sensitive_qa"
MODEL_NAME="madaptor"
EXP_NAME="baseline"
LORA="--lora"
MATRYOSHKA_DIM="768"
FLASHRAG_CONFIG="/home/thuy0050/code/FlashRAG/louis/configs/qwen8b.yaml"

sbatch /home/thuy0050/code/tevatron/src/tevatron/louis/scripts/rag/simple_rag.sh "$MODEL" "$DATA_NAME" "$MODEL_NAME" "$EXP_NAME" "$LORA" "$MATRYOSHKA_DIM" "$FLASHRAG_CONFIG"