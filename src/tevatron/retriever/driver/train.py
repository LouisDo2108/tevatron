import wandb
import logging
import os
import sys
from copy import deepcopy
from pdb import set_trace as st
from dataclasses import asdict

import torch
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import random
from pathlib import Path

from transformers.utils.import_utils import is_torch_available
from transformers.hf_argparser import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from tevatron.retriever.arguments import DataArguments, ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.collator import TrainCollator
from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer
from tevatron.retriever.modeling import DenseModel
# from tevatron.retriever.trainer import TevatronTrainer as Trainer
from tevatron.retriever.trainer import MAdaptorTrainer as Trainer

logger = logging.getLogger(__name__)

import msgspec
from msgspec.json import format

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()


def set_seed(seed: int, deterministic: bool = True):
    # Copy from transformers.trainer_utilss.set_seed with some modifications
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
        if deterministic:
            # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8 (may limit overall performance) or :4096:8 (will increase library footprint in GPU memory by approximately 24MiB). From https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

            # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
            torch.use_deterministic_algorithms(True)

            # # Enable CUDNN deterministic mode
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False


def write_json(file_path, data, jsonl=False):
    with open(file_path, "wb") as file:
        if jsonl:
            file.write(encoder.encode_lines(data))
        else:
            file.write(format(encoder.encode(data)))
    print(f"The file contains {len(data)} items.")
    print("Saved to", file_path)


def get_params_info(model):
    all_param = 0
    trainable_param = 0

    print("\nAll trainable parameters:")
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_param += param.numel()
            print(name, param.numel())
            
    print(f"trainable params: {trainable_param:,} || all params: {all_param:,} || trainable%: {trainable_param / all_param * 100:.2f}")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    default_config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.padding_side == "right":
        tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "left"

    if training_args.bf16:
        torch_dtype = torch.bfloat16
        print(f"Training in bf16")
    elif training_args.fp16:
        torch_dtype = torch.float16
        print(f"Training in fp16")
    else:
        torch_dtype = torch.float32
        print(f"Training in fp32")

    model = DenseModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        torch_dtype=default_config.torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )
    get_params_info(model)

    train_dataset = TrainDataset(data_args)
    collator = TrainCollator(data_args, tokenizer)
    
    eval_data_args = deepcopy(data_args)
    eval_data_args.dataset_path = data_args.eval_dataset_path
    eval_data_args.dataset_split = "eval"
    eval_dataset = TrainDataset(data_args)
    
    logger.info(f"Using {TrainCollator} collator and {model} model")

    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    train_dataset.set_trainer(trainer)
    eval_dataset.set_trainer(trainer)

    last_checkpoint = None
    # if os.path.isdir(training_args.output_dir):
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # if last_checkpoint:
    #     model = model.load(
    #         model_args.model_name_or_path,
    #         pooling=model_args.pooling,
    #         normalize=model_args.normalize,
    #         lora_name_or_path=model_args.lora_name_or_path,
    #         cache_dir=model_args.cache_dir,
    #         attn_implementation=model_args.attn_implementation,
    #     )
    trainer.train(resume_from_checkpoint=(last_checkpoint is not None))
    if wandb.run is not None:
        wandb.run.config.update(asdict(model_args), allow_val_change=True)
        wandb.run.config.update(asdict(data_args), allow_val_change=True)

    training_args_to_save = asdict(deepcopy(training_args))
    training_args_to_save.update(asdict(model_args))
    training_args_to_save.update(asdict(data_args))
    write_json(
        os.path.join(training_args.output_dir, "full_config.json"),
        training_args_to_save,
    )
    # trainer.save_model()
    # if trainer.is_world_process_zero():
    #     tokenizer.save_pretrained(training_args.output_dir)
    # Copy the model in the checkpoint folder to the parent folder for ease of evaluation

    src_path = max(
        Path(training_args.output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
        default=None
    )

    for f in src_path.glob("*.*"):
        trg_path = src_path.parent # gets the parent of the folder 
        f.rename(trg_path.joinpath(f.name)) # moves to parent folder.
    src_path.rmdir()


if __name__ == "__main__":
    main()
