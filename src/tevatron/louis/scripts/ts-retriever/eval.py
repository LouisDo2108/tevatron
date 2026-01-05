import argparse
from pathlib import Path
from pdb import set_trace as st
from tevatron.louis.src.configs import configs
from tevatron.louis.src.utils import run


def main():
    """Main entry for temporal retriever training and evaluation."""
    parser = argparse.ArgumentParser(description="Temporal Retriever Training Pipeline")
    parser.add_argument(
        "--model",  
        required=True,
        type=str,
        help="Model name, e.g. bge | bgem3 | contriever | gte | nomic | qwen3",
    )
    parser.add_argument("--exp_name", default="dev", type=str)
    parser.add_argument(
        "--data",
        default="temporal_nobel_prize",
        choices=[
            "temporal_nobel_prize",
            "time_sensitive_qa",
        ],
        help="Dataset name",
    )
    parser.add_argument("--enhanced_temporal", action="store_true", help="Use this flag only for temporal_nobel_prize dataset to train on our enhanced temporal queries.")
    parser.add_argument("--eval", action="store_true", help="Eval with the corresponding dev set and also save the best model with the eval loss.", default=False)
    parser.add_argument("--lora", action="store_true", help="Use LoRA for training.", default=False)
    parser.add_argument("--lora_r", default=4, type=int)
    parser.add_argument("--lora_alpha", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--eval_batch_size", default=512, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--num_neg", default=1, type=int)
    args = parser.parse_args()

    # ==== PATHS ====
    HOME = Path("/home/thuy0050")
    CODE_DIR = HOME / "code" / "tevatron" / "src" / "tevatron"
    DATA_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "data" / "third_work"
    OUTPUT_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "exp" / "tevatron"
    MODEL_NAME = "ts-retriever"

    data_name = args.data
    exp_name = args.exp_name

    eval_dataset_path = ""
    if args.eval:
        eval_dataset_path = f"--eval_dataset_path {DATA_ROOT}/temporal/temporal_nobel_prize/train/dev2.jsonl --eval_on_start True --metric_for_best_model eval_recall@1_768"
        # eval_dataset_path = f"--eval_dataset_path {DATA_ROOT}/temporal/temporal_nobel_prize/train/dev.jsonl"
    else:
        eval_dataset_path = "--save_strategy epoch --load_best_model_at_end False"

    # ==== ARGUMENTS ====
    eval_batch_size = args.eval_batch_size

    # ==== MODEL CONFIG ====
    model_name = args.model
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}")
    cfg = configs[model_name]

    backbone = cfg["checkpoint"]
    output_dir = OUTPUT_ROOT / data_name / MODEL_NAME / backbone / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_eval = ""
    if args.lora:
        lora_train = f"--lora --lora_r {args.lora_r} --lora_alpha {args.lora_alpha} --lora_target_modules all-linear"
        lora_eval = f"--lora_name_or_path {output_dir}"

    # ==== COMMANDS ====
    encode_corpus_cmd = f"""
    python {CODE_DIR}/retriever/driver/encode.py \
        --per_device_eval_batch_size {eval_batch_size} \
        --passage_max_len 512 \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --dataset_name LouisDo2108/temporal-nobel-prize \
        --dataset_config corpus \
        --dataset_path {DATA_ROOT}/temporal/{data_name}/test/corpus.jsonl \
        --encode_output_path {output_dir}/corpus_emb.pkl \
        --model_name_or_path {output_dir} \
        {lora_eval} \
        --overwrite_output_dir \
        --dataloader_num_workers 8 \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {cfg['matryoshka_dim']}
    """

    encode_query_cmd = f"""
    python {CODE_DIR}/retriever/driver/encode.py \
        --per_device_eval_batch_size {eval_batch_size} \
        --query_max_len 512 \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --encode_is_query \
        --dataset_name LouisDo2108/temporal-nobel-prize \
        --dataset_config query \
        --dataset_path {DATA_ROOT}/temporal/{data_name}/test/query.jsonl \
        --model_name_or_path {output_dir} \
        {lora_eval} \
        --encode_output_path {output_dir}/queries_emb.pkl \
        --overwrite_output_dir \
        --dataloader_num_workers 8 \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {cfg['matryoshka_dim']}
    """

    retrieval_cmd = f"""
    set -f && OMP_NUM_THREADS=16 python -m tevatron.retriever.driver.search \
        --query_reps {output_dir}/queries_emb.pkl \
        --passage_reps {output_dir}/corpus_emb.pkl \
        --depth 100 \
        --batch_size 2048 \
        --save_text \
        --save_ranking_to {output_dir}/rank.txt

    && python -m tevatron.utils.format.convert_result_to_trec \
        --input {output_dir}/rank.txt \
        --output {output_dir}/rank.trec \
        --remove_query

    && python -m pyserini.eval.trec_eval -c \
        -m recall.10,100 -m ndcg_cut.10 -M 100 \
        {DATA_ROOT}/temporal/{data_name}/test/qrel.txt \
        {output_dir}/rank.trec > {output_dir}/out.txt
    """
    # && python {CODE_DIR}/louis/beir_scripts/eval_nanobeir_with_sbert.py \
    # --model_name_or_path {output_dir} \
    # --nanobeir_datasets NQ \
    # --pooling {cfg['pooling']} \
    # --bf16 \
    # --query_prompts {cfg['query_prefix']} \
    # --corpus_prompts {cfg['passage_prefix']} \
    # --matryoshka_dim {cfg['matryoshka_dim']} > {output_dir}/out_nanobeir_nq.txt

    # ==== EXECUTION ====
    run([encode_corpus_cmd, encode_query_cmd])
    run([retrieval_cmd])

if __name__ == "__main__":
    main()
