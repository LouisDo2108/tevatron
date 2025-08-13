Note that to use another tevatron's huggingface dataset config to load a custom dataset, you need to set:
```bash
python src/tevatron/retriever/driver/encode.py \
  --output_dir=$OUTPUT_DIR/ts-retriever \
  --model_name_or_path /home/thuy0050/mg61_scratch2/thuy0050/data/third_work/ts-retriever/models/Tscontriever \
  --per_device_eval_batch_size 512 \
  --dataset_name /home/thuy0050/mg61_scratch2/thuy0050/data/third_work/tevatron/Tevatron___wikipedia-nq-corpus \
  --dataset_path /home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/test/doc.jsonl \
  --passage_max_len 512 \
  --fp16 \
  --encode_output_path /home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/ts-retriever/corpus_emb.pkl \
  --attn_implementation sdpa # Scaled Dot Product Attention, for BERT
```

- Dataset_name: must be a local path to the dataset, in order to avoid downloading the dataset from huggingface hub, otherwise, the behavior is weird, including:
    - Redownloading the dataset from huggingface hub, even if the dataset is already downloaded.
    - Does not load the custom dataset, but load the original dataset from huggingface hub.
- Dataset_path: the jsonl file to the custom dataset.