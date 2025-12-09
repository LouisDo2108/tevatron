import os
import re
import argparse
import subprocess
from pathlib import Path
from pprint import pprint, pformat
from pdb import set_trace as st
from typing import Union, List, Optional

# Compile once at module load time for speed
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_cmd(cmd: str) -> str:
    """Normalize whitespace and remove newlines in a shell command."""
    cmd = _WHITESPACE_RE.sub(" ", cmd.strip())
    cmd = cmd.replace("\n", " ")
    return cmd


def run(
    cmd: Union[str, List[str]], env: Optional[dict] = None, dry_run: bool = False
) -> None:
    """
    Run one or multiple shell commands safely.

    Args:
        cmd: A single command string or a list of commands.
        env: Optional environment variables to pass to subprocess.
        dry_run: If True, prints commands instead of executing them.
    """
    if isinstance(cmd, list):
        cmd_list = [normalize_cmd(c) for c in cmd if c.strip()]
        joined_cmd = " && ".join(cmd_list)
    else:
        joined_cmd = normalize_cmd(cmd)

    # print(f"\n>>> Running:\n{joined_cmd}\n", flush=True)

    if dry_run:
        return

    try:
        # print("Running with the following environment variables:")
        # pprint({k: v for k, v in os.environ.items()})
        
        subprocess.run(joined_cmd, shell=True, check=True, env=dict(os.environ))
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        raise


configs = {
    "bge": dict(
        checkpoint="BAAI/bge-base-en-v1.5",
        pooling="cls",
        query_prefix="'Represent this sentence for searching relevant passages: '",
        passage_prefix="''",
        query_prompts="'Represent this sentence for searching relevant passages: '",
        corpus_prompts="''",
        normalize="--normalize",
        padding_side="right",
        matryoshka_dim="768",
        matryoshka_dim_list=[768],
        temperature="0.02",
    ),
    "bgem3": dict(
        checkpoint="BAAI/bge-m3",
        pooling="cls",
        query_prefix="''",
        passage_prefix="''",
        query_prompts="''",
        corpus_prompts="''",
        normalize="--normalize",
        padding_side="right",
        matryoshka_dim="1024",
        matryoshka_dim_list=[1024],
        temperature="0.02",
    ),
    "contriever": dict(
        checkpoint="facebook/contriever",
        pooling="mean",
        query_prefix="''",
        passage_prefix="''",
        query_prompts="''",
        corpus_prompts="''",
        normalize="",
        padding_side="right",
        matryoshka_dim="768",
        matryoshka_dim_list=[768],
        temperature="0.05",
    ),
    "gte": dict(
        checkpoint="thenlper/gte-base",
        pooling="mean",
        query_prefix="''",
        passage_prefix="''",
        query_prompts="''",
        corpus_prompts="''",
        normalize="--normalize",
        padding_side="right",
        matryoshka_dim="768",
        matryoshka_dim_list=[768],
        temperature="0.02",
    ),
    "gte1.5": dict(
        checkpoint="Alibaba-NLP/gte-base-en-v1.5",
        pooling="cls",
        query_prefix="''",
        passage_prefix="''",
        query_prompts="''",
        corpus_prompts="''",
        normalize="--normalize",
        padding_side="right",
        matryoshka_dim="768",
        matryoshka_dim_list=[768],
        temperature="0.02",
    ),
    "nomic": dict(
        checkpoint="nomic-ai/nomic-embed-text-v1.5",
        pooling="mean",
        query_prefix="search_query:",
        passage_prefix="search_document:",
        query_prompts="search_query:",
        corpus_prompts="search_document:",
        normalize="--normalize",
        padding_side="right",
        matryoshka_dim="768",
        matryoshka_dim_list=[768],
        temperature="0.02",
    ),
    "qwen3": dict(
        checkpoint="Qwen/Qwen3-Embedding-0.6B",
        pooling="last",
        query_prefix="'Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery:'",
        passage_prefix="''",
        query_prompts="'Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery:'",
        corpus_prompts="''",
        normalize="--normalize",
        padding_side="left",
        matryoshka_dim="1024",
        matryoshka_dim_list=[1024],
        temperature="0.02",
    ),
}


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
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--qt", default=0.0, type=float)
    parser.add_argument("--pt", default=0.0, type=float)
    parser.add_argument("--qt_recon", default=0.0, type=float)
    parser.add_argument("--pt_recon", default=0.0, type=float)
    parser.add_argument("--temporal_dim", default=64, type=int)
    parser.add_argument("--max_temporal_length", default=16, type=int)
    parser.add_argument("--temporal", action="store_true", default=False)
    parser.add_argument("--temporal_reconstruction", action="store_true", default=False)
    parser.add_argument("--filter_false_negatives", action="store_true", default=False)
    parser.add_argument("--kl_loss", action="store_true", default=False)
    parser.add_argument("--add_cls", action="store_true", default=False)
    parser.add_argument("--detach_temporal", action="store_true", default=False)
    parser.add_argument("--distillation", action="store_true", default=False)
    parser.add_argument("--new_cls", action="store_true", default=False)
    parser.add_argument("--use_residual", action="store_true", default=False)
    parser.add_argument(
        "--matryoshka_dim_list",
        type=int,
        nargs="+",  # Accepts multiple integers
        help="List of dimensions for Matryoshka representation, e.g. --matryoshka_dim_list 256 512 768",
    )
    parser.add_argument(
        "--matryoshka_dim",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    # ==== PATHS ====
    HOME = Path("/home/thuy0050")
    CODE_DIR = HOME / "code" / "tevatron"
    DATA_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "data" / "third_work"
    OUTPUT_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "exp" / "tevatron"
    MODEL_NAME = "temporal"

    data_name = args.data
    exp_name = args.exp_name

    eval_dataset_path = ""
    if args.eval:
        if data_name == "temporal_nobel_prize":
            if args.enhanced_temporal:
                eval_dataset_path = f"--eval_dataset_path {DATA_ROOT}/temporal/temporal_nobel_prize/train/dev3.jsonl --eval_on_start True --metric_for_best_model eval_recall@1_768"
            else:
                eval_dataset_path = f"--eval_dataset_path {DATA_ROOT}/temporal/temporal_nobel_prize/train/dev.jsonl --eval_on_start True --metric_for_best_model eval_recall@1_768"
            # eval_dataset_path = f"--eval_dataset_path {DATA_ROOT}/temporal/temporal_nobel_prize/train/dev.jsonl"
        else:
            if args.enhanced_temporal:
                eval_dataset_path = f"--eval_dataset_path {DATA_ROOT}/temporal/time_sensitive_qa/development/dev.jsonl --eval_on_start True --metric_for_best_model eval_recall@1_768"
            else:
                eval_dataset_path = f"--eval_dataset_path {DATA_ROOT}/temporal/time_sensitive_qa/train/dev.jsonl --eval_on_start True --metric_for_best_model eval_recall@1_768"
    else:
        eval_dataset_path = "--save_strategy epoch --load_best_model_at_end False"
    
    dataset_path = ""
    if data_name == "temporal_nobel_prize":
        if args.enhanced_temporal:
            dataset_path = f"{DATA_ROOT}/temporal/{data_name}/train/train_enhanced_temporal_v2.jsonl"
        else:
            dataset_path = f"{DATA_ROOT}/temporal/{data_name}/train/train.jsonl"
    elif data_name == "time_sensitive_qa":
        if args.enhanced_temporal:
            dataset_path = f"{DATA_ROOT}/temporal/time_sensitive_qa/development/train.jsonl"
        else:
            dataset_path = f"{DATA_ROOT}/temporal/{data_name}/train/train.jsonl"


    # ==== ARGUMENTS ====
    batch_size = args.batch_size
    epoch = args.epoch
    num_neg = args.num_neg + 1
    grad_accum = args.gradient_accumulation_steps
    grad_ckpt = "--gradient_checkpointing" if args.gradient_checkpointing else ""
    wandb = "wandb" if args.wandb else "none"

    # ==== MODEL CONFIG ====
    model_name = args.model
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}")
    cfg = configs[model_name]

    backbone = cfg["checkpoint"]
    output_dir = OUTPUT_ROOT / data_name / MODEL_NAME / backbone / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_train = ""
    lora_eval = ""
    if args.lora:
        lora_train = f"--lora --lora_r {args.lora_r} --lora_alpha {args.lora_alpha} --lora_target_modules all-linear"
        lora_eval = f"--lora_name_or_path {output_dir}"

    filter_false_negatives = ""
    if args.filter_false_negatives:
        filter_false_negatives = "--filter_false_negatives"
    temporal = ""
    if args.temporal:
        temporal = "--temporal"
    temporal_reconstruction = ""
    if args.temporal_reconstruction:
        temporal_reconstruction = "--temporal_reconstruction"

    matryoshka_dim_list_str = ""
    if not args.matryoshka_dim_list:
        matryoshka_dim_list_str = " ".join([str(x) for x in cfg["matryoshka_dim_list"]])
    else:
        matryoshka_dim_list_str = " ".join([str(x) for x in args.matryoshka_dim_list])

    matryoshka_dim = ""
    if args.matryoshka_dim is not None:
        matryoshka_dim = str(args.matryoshka_dim)
    else:
        matryoshka_dim = str(cfg["matryoshka_dim"])

    add_cls = ""
    if args.add_cls:
        add_cls = "--add_cls"
        
    new_cls = ""
    if args.new_cls:
        new_cls = "--new_cls"
    
    use_residual = ""
    if args.use_residual:
        use_residual = "--use_residual"
        
    detach_temporal = ""
    if args.detach_temporal:
        detach_temporal = "--detach_temporal"
        
    distillation = ""
    if args.distillation:
        distillation = "--distillation"

    # ==== COMMANDS ====
    train_cmd = f"""
    ulimit -n 4096 && \
    python /home/thuy0050/code/FlashRAG/louis/index_builder.py \
        --per_device_eval_batch_size 512 \
        --query_max_len 512 \
        --pooling cls \
        --bf16 \
        --normalize \
        --attn_implementation sdpa \
        --corpus_path $DATA_ROOT_DIR/temporal/temporal_nobel_prize/test/corpus.jsonl \
        --save_dir $OUTPUT_DIR \
        --model_name_or_path $CHECKPOINT_DIR \
        --passage_prefix "" \
    python {CODE_DIR}/louis/madaptor/train_temporal.py \
        --do_train \
        --pooling {cfg['pooling']} \
        --bf16 \
        {cfg['normalize']} \
        --train_group_size {num_neg} \
        --per_device_train_batch_size {batch_size} \
        --per_device_eval_batch_size {args.eval_batch_size} \
        --learning_rate {args.lr} \
        --temperature {cfg['temperature']} \
        --logging_steps 10 \
        --num_train_epochs {epoch} \
        --gradient_accumulation_steps {grad_accum} \
        {lora_train} \
        --dataset_name {DATA_ROOT}/tevatron/Tevatron___msmarco-passage \
        --dataset_path {dataset_path} \
        {eval_dataset_path} \
        --model_name_or_path {backbone} \
        --run_name {backbone}_{exp_name} \
        --output_dir {output_dir} \
        --report_to {wandb} \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {matryoshka_dim} \
        {grad_ckpt} \
        {temporal} \
        {temporal_reconstruction} \
        --temporal_dim {args.temporal_dim} \
        --pt {args.pt} \
        --qt {args.qt} \
        --pt_recon {args.pt_recon} \
        --qt_recon {args.qt_recon} \
        --dataloader_num_workers 2 \
        {filter_false_negatives} \
        {add_cls} \
        {detach_temporal} \
        {distillation} \
        {new_cls} \
        {use_residual} \
        --matryoshka_dim_list {matryoshka_dim_list_str} \
        > {output_dir}/train_log.txt
    """

    encode_corpus_cmd = f"""
    python {CODE_DIR}/src/tevatron/retriever/driver/encode.py \
        --per_device_eval_batch_size {args.eval_batch_size} \
        --passage_max_len 512 \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --dataset_name LouisDo2108/temporal-nobel-prize \
        --dataset_config corpus \
        --dataset_path {DATA_ROOT}/temporal/{data_name}/test/corpus.jsonl \
        --encode_output_path {output_dir}/corpus_emb_{matryoshka_dim}.pkl \
        --model_name_or_path {output_dir} \
        {lora_eval} \
        --overwrite_output_dir \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {matryoshka_dim}
    """
    encode_query_cmd = f"""
    python {CODE_DIR}/src/tevatron/retriever/driver/encode.py \
        --per_device_eval_batch_size {args.eval_batch_size} \
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
        --encode_output_path {output_dir}/queries_emb_{matryoshka_dim}.pkl \
        --overwrite_output_dir \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {matryoshka_dim}
    """

    retrieval_cmd = f"""
    set -f && OMP_NUM_THREADS=16 python -m tevatron.retriever.driver.search \
        --query_reps {output_dir}/queries_emb_{matryoshka_dim}.pkl \
        --passage_reps {output_dir}/corpus_emb_{matryoshka_dim}.pkl \
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
        {output_dir}/rank.trec > {output_dir}/out_{matryoshka_dim}.txt
    """
    # python {CODE_DIR}/louis/beir_scripts/eval_nanobeir_with_sbert.py \
    #     --model_name_or_path {output_dir} \
    #     --nanobeir_datasets NQ \
    #     --pooling {cfg['pooling']} \
    #     --bf16 \
    #     --query_prompts {cfg['query_prefix']} \
    #     --corpus_prompts {cfg['passage_prefix']} \
    #     --matryoshka_dim {matryoshka_dim} > {output_dir}/out_nanobeir_nq_{matryoshka_dim}.txt

    # ==== EXECUTION ====
    run([encode_corpus_cmd, encode_query_cmd])
    run([retrieval_cmd])


if __name__ == "__main__":
    main()
