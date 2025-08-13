import wandb
import logging
import os
import sys
from copy import deepcopy
from pdb import set_trace as st
from dataclasses import asdict

from utils import init, write_json, get_params_info
import torch
from transformers import AutoTokenizer

from madaptor import NaiveTemporalv4 as Model
from dataset import NaiveTemporalDataset as TrainDataset
from tevatron.retriever.dataset import TrainDataset as EvalDataset
from collator import NaiveTemporalv3Collator as TrainCollator
from trainer import MAdaptorTrainer as Trainer

from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer


def main():
    model_args, data_args, training_args = init()

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

    model = Model.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        # torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    train_dataset = TrainDataset(data_args)

    eval_data_args = deepcopy(data_args)
    eval_data_args.dataset_path = data_args.eval_dataset_path

    eval_dataset = EvalDataset(eval_data_args)
    collator = TrainCollator(data_args, tokenizer)

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
    get_params_info(model)
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
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
