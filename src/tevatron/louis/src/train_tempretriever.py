import logging
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pdb import set_trace as st
import random

import wandb
from tevatron.louis.src.collator import TempRetrieverCollator
from tevatron.louis.src.dataset import TemporalDataset
from tevatron.louis.src.models import TempRetriever
from tevatron.louis.src.trainer import MAdaptorTrainer as Trainer
from transformers import AutoConfig, AutoTokenizer
from tevatron.louis.src.utils import get_params_info, init, write_json

from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer
from tevatron.retriever.arguments import DataArguments
from datasets import load_dataset

from sutime import SUTime

def init_sutime():
    global sutime_instance
    sutime_instance = SUTime(mark_time_ranges=True, include_range=True)
    return sutime_instance

logger = logging.getLogger(__name__)


class TemporalDataset(TrainDataset):

    def __init__(
        self,
        data_args: DataArguments,
        trainer=None,
        dataset_name=None,
        corpus_name=None,
        dataset_path=None,
        corpus_path=None,
        corpus_assets_path=None,
    ):
        self.data_args = data_args
        self.trainer = trainer

        # Load training data
        self.train_data = load_dataset(
            self.data_args.dataset_name if dataset_name is None else dataset_name,
            self.data_args.dataset_config,
            data_files=(
                self.data_args.dataset_path if dataset_path is None else dataset_path
            ),
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            num_proc=self.data_args.num_proc,
        )   

        # Load corpus if provided
        if self.data_args.corpus_name is None and corpus_name is None:
            self.corpus = None
        else:
            self.corpus = load_dataset(
                self.data_args.corpus_name if corpus_name is None else corpus_name,
                self.data_args.corpus_config,
                data_files=(
                    self.data_args.corpus_path if corpus_path is None else corpus_path
                ),
                split=self.data_args.corpus_split,
                cache_dir=self.data_args.dataset_cache_dir,
                num_proc=self.data_args.num_proc,
            )

        self.passage_prefix = self.data_args.passage_prefix.replace("\\n", "\n").strip()
        if self.passage_prefix != "":
            self.passage_prefix += " "
            
        self.query_prefix = self.data_args.query_prefix.replace("\\n", "\n").strip()
        if self.query_prefix != "":
            self.query_prefix += " "

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        """
        Note that for temporal dataset, since the "query" here is actually the passage and positive/negative "passages" are actually the "query"; 
        *** during training, we should add the passage_prefix to the "query" and query_prefix to the "passages"
        *** during evaluation/inference, they are normal.
        *** This is by swapping query_prefix and passage_prefix in the data_args.
        """

        query_text = group.get('query', "")  # type: ignore
        query_temporal = group.get("temporal", "")  # type: ignore

        formatted_query = (
            self.passage_prefix + query_text
        )  # reversed role of passage_prefix and query_prefix

        formatted_documents = []

        # Select positive document
        selected_positive = group["positive_passages"][
            (_hashed_seed + epoch) % len(group["positive_passages"])
        ]
        positive_text = (
            selected_positive["title"] + " " + selected_positive["text"]
            if "title" in selected_positive
            else selected_positive["text"]
        )
        formatted_documents.append(
            (
                self.query_prefix + positive_text, # reversed role of passage_prefix and query_prefix
                selected_positive.get('temporal', ""),
                selected_positive.get('temporal_query_type', ""), 
                selected_positive.get('allen_relation', ""),
            )
        )

        """
        The positive passage here refers to the "temporal question" that is related to the document. Let's stick with the name positive query.
        Get the docid corresponds to the positive query.
        Assuming that there is only one positive docid.
        """

        # Select negative documents
        negative_size = self.data_args.train_group_size - 1
        
        num_negatives = len(group.get("negative_passages", []))
        
        if num_negatives > 0 and num_negatives < negative_size:
            selected_negatives = random.choices(
                group["negative_passages"], k=negative_size
            )
        elif num_negatives == 0 or self.data_args.train_group_size == 1:
            selected_negatives = []
        else:
            offset = epoch * negative_size % num_negatives
            selected_negatives = list(group["negative_passages"])
            random.Random(_hashed_seed).shuffle(selected_negatives)
            selected_negatives = selected_negatives * 2
            selected_negatives = selected_negatives[offset : offset + negative_size]

        for negative in selected_negatives:
            negative_text = (
                negative["title"] + " " + negative["text"]
                if "title" in negative
                else negative["text"]
            )
            formatted_documents.append(
                (
                    self.query_prefix + negative_text,
                    negative.get("temporal", ""),
                    negative.get("temporal_query_type", ""),
                    negative.get("allen_relation", ""),
                )
            )
        
        # if query_temporal == "":
        #     query_temporal = sutime_instance.parse(content_text)
        #     content_temporal = ", ".join([t["text"] for t in content_temporal])

        #     query_temporal = ""sutime.parse()
        
        return formatted_query, query_temporal, formatted_documents


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


def main():
    model_args, data_args, training_args = init()

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

    model = TempRetriever.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        dtype=default_config.dtype,
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
        # eval_data_args.dataset_split = "eval"
        eval_data_args.train_group_size = 2
        eval_dataset = TrainDataset(eval_data_args)
    else:
        training_args.eval_strategy = "no"
        training_args.save_strategy = "epoch"
        training_args.load_best_model_at_end = False

    # logger.info(f"Using {collator} collator and {model} model")

    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.tempretriever = True
    train_dataset.set_trainer(trainer)

    # if eval_dataset is not None:
    #     eval_dataset.set_trainer(trainer)

    last_checkpoint = None
    trainer.train(resume_from_checkpoint=(last_checkpoint is not None))
    # if wandb.run is not None:
    #     wandb.run.config.update(asdict(model_args), allow_val_change=True)
    #     wandb.run.config.update(asdict(data_args), allow_val_change=True)

    training_args_to_save = asdict(deepcopy(training_args))
    training_args_to_save.update(asdict(model_args))
    training_args_to_save.update(asdict(data_args))
    write_json(
        os.path.join(training_args.output_dir, "full_config.json"),
        training_args_to_save,
    )
    # from safetensors import safe_open
    # tensors = {}
    # with safe_open("/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/tempretriever/facebook/contriever/baseline-our-dataset/checkpoint-10/model.safetensors", framework="pt", device="cpu") as f:
    #     for key in f.keys():
    #         tensors[key] = f.get_tensor(key)
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
