# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from pdb import set_trace as st
import time
from tqdm.auto import tqdm

import re
import os
import sys
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset, default_collate
from dataclasses import dataclass
from datasets import load_dataset

from transformers import AutoTokenizer, HfArgumentParser, set_seed
from tevatron.retriever.arguments import DataArguments, ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.arguments import DataArguments

import msgspec
from tqdm.auto import tqdm

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()


def read_json(file_path, jsonl=False):
    with open(file_path, "wb") as file:
        data = file.read()
    if jsonl:
        output = decoder.decode_lines(data)
    else:
        output = decoder.decode(data)

    print(f"The file is of type: {type(output)}")
    print(f"The file contains {len(output)} items.")
    return output


def write_json(file_path, data, jsonl=False):
    with open(file_path, "wb") as file:
        if jsonl:
            file.write(encoder.encode_lines(data))
        else:
            file.write(encoder.encode(data))
    print(f"The file contains {len(data)} items.")
    print("Saved to", file_path)

# In specific cases where historical facts are universally known and confidently implied (e.g., 'after 1945' or 'from 1946' for events following World War II), you may include them.
# Define task description
task_description = """
You must extract relevant yet concise temporal information/expressions that are strongly connected to the content of the document. The goal is to improve the performance of temporal information retrieval and aid complex tasks such as temporal reasoning. 
These expressions include:
- Explicit dates (e.g., "March 2023", "21 July 2011")
- Implicit cues (e.g., "recently", "during his presidency", "after a major power outage")
- Relative temporal expressions (e.g., "last week", "that summer") should be included only when clearly anchored to an explicit date or year in the same or preceding sentence. For example, in "He played until 2011. That summer he transferred to France."), you should include "until 2011", "that summer", and/or "summer 2011".
- Event-based references (e.g., "2024 Olympics", "2009 Georgian Women’s Championship", "World Cup 2018", etc.)
- Durations or time spans (e.g., "from 2013 to 2014", "since 1990", "until 2000").
Pay close attention to the temporal relationships in the text, such as "after", "before", "during", "since", and "until". These relationships are crucial for understanding the temporal context of events.
"""

# Define demonstration examples
demo_examples = [
    (
        "What’s the capital of Han dynasty from 8 to 9?",
        "from 8 to 9",
    ),
    (
        "Ellen Kooi worked in  which location from 1993 to 1994?",
        "from 1993 to 1994",
    ),
    (
        "What operated EMD F45 after Nov 1970?",
        "after Nov 1970",
    ),
    (
        "What was the official name of Patrol Squadron 4 (United States Navy) before Mar 1941?",
        "before Mar 1941",
    ),
    (
        "Janet Napolitano took which position as of 2003",
        "as of 2003",
    ),
    (
        "What happened in Poland after World War II and before 1960?",
        "after World War II, before 1960",
    ),
]

# Format demo examples
demo = ""
for doc, output in demo_examples:
    demo += f'Document: "{doc}" Output: {output}'

# Define input template
input_template = 'Input:  Document: "{}". Output: {}'

# Assemble final prompt template
# Inspired by hipporag
prompt_template = f"""
You are an expert in preprocessing data. {task_description}
The input template is as follows: {input_template}. 
Some examples for your reference:
{demo.strip()}
Now, you should output the temporal expressions in the given documents. Only output the temporal expressions.
"""


class LLMDataset(Dataset):

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
        """
        Dataset for encoding documents with LLM.
        :param data_args: DataArguments
        """
        self.data_args = data_args
        self.trainer = trainer

        # Load training data
        self.train_data = load_dataset(
            self.data_args.dataset_name if dataset_name is None else dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path if dataset_path is None else dataset_path,
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

        # for video we use assets_path to load the video
        self.corpus_assets_path = (
            corpus_assets_path
            if corpus_assets_path is not None
            else self.data_args.assets_path
        )

        # create a map between docid and index
        self.docid_to_index = {}
        if self.corpus is not None:
            corpus_ids = self.corpus.select_columns(["docid"])
            docids = corpus_ids["docid"]
            self.docid_to_index = {
                docid: index for index, docid in enumerate(tqdm(docids))
            }

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        content = self.train_data[item]
        return content


def quick_text_normalize(text):
    # Flatten whitespace (includes \n, \t, multiple spaces)
    text = " ".join(text.split())

    # Fix punctuation spacing artifacts
    text = text.replace(" . ", ". ").replace(" , ", ", ")

    # De-space character-level corrupted words, e.g., "P e r t h" → "Perth"
    text = re.sub(
        r"(?<=\b)(?:[A-Za-z]\s)+(?:[A-Za-z])(?=\b)", 
        lambda m: m.group(0).replace(" ", ""), 
        text
    )

    # Normalize spacing around commas
    text = ",".join([part.strip() for part in text.split(",")])

    return text


def quick_text_normalize(text):
    # Flatten whitespace (includes \n, \t, multiple spaces)
    text = " ".join(text.split())

    # Fix punctuation spacing artifacts
    text = text.replace(" . ", ". ").replace(" , ", ", ")

    # De-space character-level corrupted words, e.g., "P e r t h" → "Perth"
    text = re.sub(
        r"(?<=\b)(?:[A-Za-z]\s)+(?:[A-Za-z])(?=\b)",
        lambda m: m.group(0).replace(" ", ""),
        text,
    )

    # # Normalize spacing around commas
    # text = ",".join([part.strip() for part in text.split(",")])

    return text


def llm_collate(batch):
    """
    Collate function for encoding.
    :param features: list of (id, text, image) tuples
    but in this case, it's just image is None
    """
    query = {i['query_id']: i['query'] for i in batch}
    positive = [i["positive_passages"] for i in batch]
    negative = [i["negative_passages"] for i in batch]

    positive_id = [(qid, x["docid"]) for qid, i in zip(query.keys(), positive) for x in i]
    positive_text = [x["text"] for i in positive for x in i]
    negative_id = [(qid, x["docid"]) for qid, i in zip(query.keys(), negative) for x in i]
    negative_text = [x["text"] for i in negative for x in i]

    messages_positive = [
        [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": f'Document: "{doc}" Output: '},
        ]
        for doc in positive_text
    ]

    messages_negative = [
        [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": f'Document: "{doc}" Output: '},
        ]
        for doc in negative_text
    ]
    return (
        query,
        positive_id,
        positive_text,
        messages_positive,
        negative_id,
        negative_text,
        messages_negative,
    )


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

    batch_size = training_args.per_device_train_batch_size

    dataset = LLMDataset(data_args)
    # collator = UnsupervisedMAdaptorCollator(data_args, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=llm_collate,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    sampling_params = SamplingParams(max_tokens=512)

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_num_batched_tokens=128 * batch_size,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        generation_config="auto",
        max_model_len=10000,  # Limit context window
        max_num_seqs=batch_size,  # Limit batch size
        gpu_memory_utilization=0.95,
        disable_cascade_attn=True,  # Avoid gibberish output due to batch inference
        seed=42,
        # model="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    )

    temporal_jsonl = []

    for i in tqdm(loader):
        (
            query,
            positive_id,
            positive_text,
            messages_positive,
            negative_id,
            negative_text,
            messages_negative,
        ) = i

        outputs_positive = llm.chat(messages=messages_positive, sampling_params=sampling_params)
        outputs_positive = [x.outputs[0].text for x in outputs_positive]

        outputs_negative = llm.chat(messages=messages_negative, sampling_params=sampling_params)
        outputs_negative = [x.outputs[0].text for x in outputs_negative]

        temp_dict = {}
        for (qid, docid), text, temporal in zip(positive_id, positive_text, outputs_positive):
            if temp_dict.get(qid, None) is None:
                temp_dict[qid] = {}
                temp_dict[qid]['positive_passages'] = []
            temp_dict[qid]["positive_passages"].append(
                {
                    "docid": docid,
                    "text": text,
                    "temporal": temporal.split(","),
                }
            )

        for (qid, docid), text, temporal in zip(negative_id, negative_text, outputs_negative):
            if temp_dict[qid].get("negative_passages", None) is None:
                temp_dict[qid]["negative_passages"] = []
            temp_dict[qid]["negative_passages"].append(
                {
                    "docid": docid,
                    "text": text,
                    "temporal": temporal.split(","),
                }
            )

        for qid, content in temp_dict.items():
            temporal_jsonl.append(
                {
                    "query_id": qid,
                    "query": quick_text_normalize(query[qid]),
                    "positive_passages": content["positive_passages"],
                    "negative_passages": content["negative_passages"],
                }
            )

    write_json(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/temporal_nobel_prize/train/train_temporal_v2.jsonl",
        temporal_jsonl,
        jsonl=True,
    )

    del llm
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
