import logging
import os
import sys
from copy import deepcopy
from pdb import set_trace as st
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, HfArgumentParser, set_seed

from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.collator import TrainCollator
from tevatron.retriever.arguments import DataArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.arguments import DataArguments


class UnsupervisedMAdaptorDataset(TrainDataset):
    def __getitem__(self, item):
        content = self.train_data[item]
        content_id = content["docid"]
        content_text = content.get("text", "")

        dummy_query = []
        return dummy_query, (content_id, content_text)


def main():
    parser = HfArgumentParser(DataArguments, TrainingArguments)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        data_args, training_args = parser.parse_args_into_dataclasses()
        data_args: DataArguments
        training_args: TrainingArguments

    set_seed(training_args.seed)

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
    tokenizer.padding_side = "right"

    dataset = UnsupervisedMAdaptorDataset(data_args)
    collator = TrainCollator(data_args, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
