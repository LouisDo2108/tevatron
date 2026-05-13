import logging
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pdb import set_trace as st

import wandb
from transformers import AutoConfig, AutoTokenizer

from tevatron.louis.src.collator import TemporalReconCollator, TempRetrieverCollator
from tevatron.louis.src.dataset import TemporalDataset, TempRetrieverTemporalDataset
from tevatron.louis.src.models import (
    DenseModel,
    NaiveTemporal,
    MAdaptor,
    MRL,
    TemporalProjectorReconstruction,
    TempRetriever,
)
from tevatron.louis.src.trainer import MAdaptorTrainer as Trainer
# from tevatron.retriever.trainer import Trainer
from tevatron.louis.src.utils import get_params_info, init, write_json
from tevatron.retriever.collator import TrainCollator
from tevatron.retriever.dataset import TrainDataset

logger = logging.getLogger(__name__)


def get_tokenizer(model_args, data_args):
    try:
        default_config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    except Exception as e:
        print("Cannot get the model name, use adapter_config.json instead.")
        import json
        with open(os.path.join(model_args.model_name_or_path, "adapter_config.json")) as f:
            temp_config = json.load(f)
        default_config = AutoConfig.from_pretrained(temp_config["base_model_name_or_path"], trust_remote_code=True)
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
    return default_config, tokenizer


MODEL_CLS_DICT = {
    "tmrl": TemporalProjectorReconstruction,
    "mrl": MRL,
    "madaptor": MAdaptor,
    "tempretriever": TempRetriever,
    "ts-retriever": NaiveTemporal,
    "tsm": NaiveTemporal,
}


def main():
    model_args, data_args, training_args = init()

    default_config, tokenizer = get_tokenizer(model_args, data_args)

    if training_args.method_name == "tempretriever":
        train_dataset = TempRetrieverTemporalDataset(data_args)
        collator = TempRetrieverCollator(data_args, tokenizer)
    elif training_args.method_name == "tmrl":
        train_dataset = TemporalDataset(data_args)
        collator = TemporalReconCollator(
            data_args, tokenizer, max_temporal_length=training_args.max_temporal_length
        )
    else:
        train_dataset = TrainDataset(data_args)
        collator = TrainCollator(data_args, tokenizer)

    eval_dataset = None
    if data_args.eval_dataset_path is not None:
        eval_data_args = deepcopy(data_args)
        eval_data_args.dataset_path = data_args.eval_dataset_path
        eval_data_args.train_group_size = 2
        eval_dataset = TrainDataset(eval_data_args)
    else:
        training_args.eval_strategy = "no"
        training_args.save_strategy = "epoch"
        training_args.load_best_model_at_end = False

    method_name = training_args.method_name
    MODEL_CLS = MODEL_CLS_DICT[method_name]

    model = MODEL_CLS.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        dtype=default_config.dtype,
    )
    if method_name == "madaptor":
        for k, v in model.named_parameters():
            if "adaptor" not in k:
                v.requires_grad = False
    elif method_name == "tempretriever":
        for k, v in model.named_parameters():
            v.requires_grad = True

    if method_name == "tempretriever":
        def get_params_info_v2(model):
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
        get_params_info_v2(model)
    else:
        get_params_info(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    if method_name == "madaptor":
        try:
            if hasattr(trainer, 'madaptor'):
                trainer.madaptor = True
        except Exception as e:
            print("Cannot set madaptor attribute to trainer.")
    elif method_name == "tempretriever":
        try:
            if hasattr(trainer, 'tempretriever'):
                trainer.tempretriever = True
        except Exception as e:
            print("Cannot set tempretriever attribute to trainer.")

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
