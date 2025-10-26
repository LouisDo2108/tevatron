import logging
import argparse
from sentence_transformers import models, SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator
from pdb import set_trace as st
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def main():
    parser = argparse.ArgumentParser(description="Run NanoBEIR evaluation.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the model or model checkpoint.",
    )
    # parser.add_argument("--lora_name_or_path", default=None)
    parser.add_argument(
        "--nanobeir_datasets", type=str, nargs="+", default=["NQ"], help="List of dataset names."
    )
    parser.add_argument("--pooling", type=str, default="mean", help='Pooling can either be "cls", "lasttoken", "max", "mean"')
    parser.add_argument(
        "--query_prompts",
        type=str,
        default="",
    )
    parser.add_argument(
        "--corpus_prompts",
        type=str,
        default=""
    )
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--bf16", default=True, action="store_true")
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length."
    )
    parser.add_argument(
        "--matryoshka_dim", type=int, default=768, help="Matryoshka dimension"
    )

    args = parser.parse_args()
    logger.info("Start NanoBEIR evaluation")

    torch_dtype = "bf16" if args.bf16 else "fp16"
    logger.info(f"Inference in {torch_dtype}")

    if args.pooling == "avg":
        args.pooling = "mean"
    if args.pooling == "last":
        args.pooling = "lasttoken"
    logger.info(f"Pooling method: {args.pooling}")

    # Load the transformer and SentenceTransformer model
    transformer_model = models.Transformer.load(
        args.model_name_or_path, 
        trust_remote_code=True
    )

    # Make sure that the current model is PEFT
    if (Path(args.model_name_or_path) / "adapter_config.json").exists():
        is_there_lora = False
        for k, _ in transformer_model.named_parameters():
            if "lora" in k:
                is_there_lora = True
                break
        # This is for nomic v1.5
        if not is_there_lora:
            transformer_model.auto_model.load_adapter(args.model_name_or_path)

    pooling_model = models.Pooling(
        transformer_model.get_word_embedding_dimension(), pooling_mode=args.pooling
    )
    normalize_model = models.Normalize()

    # Set up the SentenceTransformer model
    try:
        model = SentenceTransformer(
            modules=[transformer_model, pooling_model, normalize_model],
            model_kwargs={
                "torch_dtype": torch_dtype,
                "max_seq_length": args.max_seq_length,
                "attn_implementation": "flash_attention_2"
            },
            tokenizer_kwargs={
                "padding_side": "left" if "qwen" in args.model_name_or_path else None
            },
            trust_remote_code=True,
        )
    except Exception as e:
        model = SentenceTransformer(
            modules=[transformer_model, pooling_model, normalize_model],
            model_kwargs={
                "torch_dtype": torch_dtype,
                "max_seq_length": args.max_seq_length,
                "attn_implementation": "sdpa"
            },
            tokenizer_kwargs={
                "padding_side": "left" if "qwen" in args.model_name_or_path else None
            },
            trust_remote_code=True,
        )

    args.query_prompts =args.query_prompts.replace("\\n", "\n")
    args.corpus_prompts = args.corpus_prompts.replace("\\n", "\n")

    # Set up and run the evaluator
    evaluator = NanoBEIREvaluator(
        dataset_names=args.nanobeir_datasets,
        query_prompts=args.query_prompts if args.query_prompts else None,
        corpus_prompts=args.corpus_prompts if args.corpus_prompts else None,
        truncate_dim=args.matryoshka_dim,
    )
    results = evaluator(model)
    print(f"{evaluator.primary_metric}: {results[evaluator.primary_metric]:.3f}")


if __name__ == "__main__":
    main()
