import logging
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pdb import set_trace as st

import numpy as np
import wandb
from collator import TempRetrieverCollator
from dataset import TemporalDataset
from madaptor import TempRetriever
from trainer import MAdaptorTrainer as Trainer
from transformers import AutoConfig, AutoTokenizer
from utils import get_params_info, init, write_json

from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer

logger = logging.getLogger(__name__)


# collator_dict = {
#     "tevatron_standard": TevatronCollator,
#     "temporal_as_sentence": TemporalAsSentenceCollator,
#     "extracted_temporal": ExtractedTemporalCollator,
#     "extracted_temporal_with_reconstruction": ExtractedTemporalWithReconstructionCollator,
# }


def get_params_info(model):
    all_param = 0
    trainable_param = 0

    print("\nAll trainable parameters:")
    for name, param in model.named_parameters():
        all_param += param.numel()

        if param.requires_grad:
            trainable_param += param.numel()
            print(name, param.numel())

    print(
        f"trainable params: {trainable_param:,} || all params: {all_param:,} || trainable%: {trainable_param / all_param * 100:.2f}"
    )


# def select_collators_and_models(model_args, data_args, training_args):
    
#     # Check if this is normal TS-Retriever style training
#     if not training_args.matryoshka:
#         return collator_dict["tevatron_standard"], DenseModel
    
#     if training_args.temporal:
#         if training_args.temporal_as_sentence:
#             return collator_dict["temporal_as_sentence"], NaiveTemporal
#         if training_args.extracted_temporal:
#             return collator_dict["extracted_temporal"], NaiveTemporalProjector
#         if training_args.temporal_reconstruction:
#             return collator_dict["extracted_temporal_with_reconstruction"], NaiveTemporalProjectorReconstruction
#         # Default case for temporal
#         return collator_dict["extracted_temporal"], NaiveTemporalProjector
#     else:
#         # Only semantic matryoshka
#         return collator_dict["tevatron_standard"], NaiveTemporal


def main():
    model_args, data_args, training_args = init()

    # TrainCollator, Model = select_collators_and_models(model_args, data_args, training_args)

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

    # model = Model.build(
    #     model_args,
    #     training_args,
    #     torch_dtype=default_config.torch_dtype,
    #     cache_dir=model_args.cache_dir,
    #     attn_implementation=model_args.attn_implementation,
    # )

    # train_dataset = TrainDataset(data_args)
    # collator = TrainCollator(data_args, tokenizer)

    model = TempRetriever.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        torch_dtype=default_config.torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )
    for k, v in model.named_parameters():
        v.requires_grad = True

    get_params_info(model)

    train_dataset = TemporalDataset(data_args)
    collator = TempRetrieverCollator(data_args, tokenizer)

    eval_dataset = None
    if data_args.eval_dataset_path is not None:
        eval_data_args = deepcopy(data_args)
        eval_data_args.dataset_path = data_args.eval_dataset_path
        eval_data_args.dataset_split = "eval"
        eval_dataset = TrainDataset(data_args)
    else:
        training_args.eval_strategy = "no"
        training_args.save_strategy = "epoch"
        training_args.load_best_model_at_end = False

    logger.info(f"Using {collator} collator and {model} model")

    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    train_dataset.set_trainer(trainer)
    if eval_dataset is not None:
        eval_dataset.set_trainer(trainer)

    last_checkpoint = None
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
