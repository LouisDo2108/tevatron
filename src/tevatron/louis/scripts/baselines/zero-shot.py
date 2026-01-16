import argparse
from pathlib import Path
from pdb import set_trace as st
from tevatron.louis.src.configs import configs
from tevatron.louis.src.utils import run


def main():
    """Run zero-shot encoding and retrieval for temporal datasets."""
    parser = argparse.ArgumentParser(
        description="Zero-Shot Temporal Retriever Evaluation"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Model name, e.g. bge | bgem3 | contriever | gte | nomic | qwen3",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--data",
        default="temporal_nobel_prize",
        choices=[
            "temporal_nobel_prize",
            "time_sensitive_qa",
        ],
        help="Dataset name",
    )
    args = parser.parse_args()

    # ==== PATHS ====
    HOME = Path("/home/thuy0050")
    CODE_DIR = HOME / "code" / "tevatron" / "src" / "tevatron"
    DATA_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "data" / "third_work"
    OUTPUT_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "exp" / "tevatron"

    data_name = args.data
    MODEL_NAME = "zero-shot"

    # ==== MODEL CONFIGS ====
    # If you want to run all models in `configs`, loop below.
    # Otherwise, you could also just do `cfg = configs[args.model]`.
    # for model_key, cfg in configs.items():
    cfg = configs[args.model]
    eval_batch_size = args.eval_batch_size
    backbone = cfg["checkpoint"]
    output_dir = OUTPUT_ROOT / data_name / MODEL_NAME / backbone
    output_dir.mkdir(parents=True, exist_ok=True)

    # === COMMANDS ===
    encode_corpus_cmd = f"""
    python {CODE_DIR}/retriever/driver/encode.py \
        --per_device_eval_batch_size {eval_batch_size} \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --dataset_name LouisDo2108/temporal-nobel-prize \
        --dataset_config corpus \
        --dataset_path {DATA_ROOT}/temporal/{data_name}/test/corpus.{"jsonl" if "nobel" in data_name else "parquet"} \
        --encode_output_path {output_dir}/corpus_emb.pkl \
        --model_name_or_path {cfg['checkpoint']} \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {cfg['matryoshka_dim']}
    """

    encode_query_cmd = f"""
    python {CODE_DIR}/retriever/driver/encode.py \
        --per_device_eval_batch_size {eval_batch_size} \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --encode_is_query \
        --dataset_name LouisDo2108/temporal-nobel-prize \
        --dataset_config query \
        --dataset_path {DATA_ROOT}/temporal/{data_name}/test/query.{"jsonl" if "nobel" in data_name else "parquet"} \
        --model_name_or_path {cfg['checkpoint']} \
        --encode_output_path {output_dir}/queries_emb.pkl \
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
    # --model_name_or_path {cfg['checkpoint']} \
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
