# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from pdb import set_trace as st
import time
from tqdm.auto import tqdm

import logging
import os
import sys
from copy import deepcopy
from pdb import set_trace as st
from torch.utils.data import DataLoader, Dataset, default_collate
from dataclasses import dataclass
from datasets import load_dataset

from transformers import AutoTokenizer, HfArgumentParser, set_seed
from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.collator import TrainCollator, EncodeCollator
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
- Durations or time spans (e.g., "from 2013 to 2014", "since 1990", "until 2000")
"""
#  Always include the original expression and additionally the normalized form (e.g., "summer 2011") only if the anchor is unambiguous. Otherwise, omit the normalized form. Do not add any explanations.
# If the document belongs to a diachronic corpus (spanning long periods), perform timeline summarization like a professional historian or journalist.
# If it belongs to a synchronous corpus (snapshot in time), extract all relevant temporal expressions for that context.

# Define demonstration examples
demo_examples = [
    (
        "The history of Poland from 1945 to 1989 spans the period of Marxist–Leninist regime in Poland after the end of World War II.",
        "from 1945 to 1989, Marxist–Leninist regime in Poland, after the end of World War II",
    ),
    (
        "He signed a two-year contract with League One side Crewe Alexandra in June 2013 after manager Steve Davis paid Macclesfield an undisclosed fee. However, he played just five games for the Railwaymen, being sent on two loan spells to Lincoln City, before returning to Macclesfield Town in February 2015. He then played for newly relegated Notts County in League Two for two seasons. In July 2017, Audel joined Barrow, moving to Welling United a year later.",
        "two-year contract, in June 2013, in February 2015, for two seasons, in July 2017, a year later",
    ),
    (
        "A week later Rangers made an improved offer ... On 21 July 2011, a bid of £1.5m from Rangers was accepted by Hearts.",
        "a week later, on 21 July 2011",
    ),
    (
        "We design ablation experiments for the proposed linking method, graph construction method, and triple filtering method, with the results reported in Table 4.",
        "",
    ),
    (
        "He won his first league title in 2005 and was the clubs first-choice goalkeeper up until 2011, and was the clubs first-choice goalkeeper up until 2011, making over 200 appearances for América. That summer Ochoa was transferred to Ajaccio in France. He spent three seasons with the club until their relegation from Ligue 1. In 2014, Ochoa joined Málaga but failed to establish himself in the team. In July 2016, he joined Granada on a season-long loan. In July 2017, he joined Standard Liège.",
        "2005, until 2011, that summer, summer 2011, three seasons, until relegation from Ligue 1, in 2014, in July 2016, in July 2017",
    ),
    (
        "What happened in Poland after World War II and before 1960?",
        "after World War II, after 1945, from 1946, before 1960",
    ),
]

# Format demo examples
demo = ""
for doc, output in demo_examples:
    demo += f'Document: "{doc}" Output: {output}'

# Define input template
input_template = 'Previous output: "<PREVIOUS OUTPUT STARTS FROM HERE>" Output: <YOUR OUTPUT STARTS FROM HERE>'

# Assemble final prompt template
# Inspired by hipporag
prompt_template = f"""
You were previously tasked with: {task_description}
You are now required to refine your previous output. I do not want to see any \\n\\n or \\n in your output and no explanations, such as "note that", "note", for any temporal expressions. I want a clean comma-seperated list of temporal information for each document. If the previous output is empty, leave it empty. You are only allowed to output words that are provided.
The input template is as follows: {input_template}.
"""

# Sample prompts.
documents = [
    "Following graduation , Liina Tennosaar began a two year engagement at the Vanemuine theatre in Tartu , ending in 1988 . From 1988 until 1993 , she was engaged as an actress at the Endla Theatre in Pärnu , and from 1996 until 2001 , she was engaged at the Vanalinnastuudio in Tallinn . Since 2001 , she has been a freelance actor . She has appeared in productions in many other theatres throughout Estonia , including the Ugala theatre , the Tallinna Kammerteater , the Vannalinnastudio , and others .",
    "Later in 1963 , Pelican Island was designated as a National Historic Landmark by the Secretary of the Interior because of its status as the first federal area set aside specifically to protect wildlife .",
]

prompts = [
    {"role": "system", "content": prompt_template},
    {
        "role": "user",
        "content": f'Previous output: "{documents[0]}" Output: ',
    },
]

generating_prompts = [prompt_template + f' Document: "{prompt}" Output: ' for prompt in prompts]


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
        content_chunkid = content["chunkid"]
        content_docid = content["docid"]
        content_text = content.get("temporal", "")

        return content_chunkid, content_docid, content_text

import re


def quick_text_normalize(text):
    # Remove line breaks
    text = text.replace("\n", "")

    # Fix common punctuation spacing issues
    text = text.replace(" . ", ". ")
    text = text.replace(" , ", ", ")

    # Remove leading/trailing whitespace and fix inner spacing
    text = " ".join(text.strip().split())

    # De-space character-level corrupted words (e.g., "P e r t h" → "Perth")
    def merge_spaced_letters(match):
        return match.group(0).replace(" ", "")

    text = re.sub(
        r"(?<=\b)(?:[A-Za-z]\s)+(?:[A-Za-z])(?=\b)", merge_spaced_letters, text
    )
    text = ",".join([x.strip() for x in text.split(",")]) # Normalize the delimiter

    return text


def llm_collate(batch):
    """
    Collate function for encoding.
    :param features: list of (id, text, image) tuples
    but in this case, it's just image is None
    """
    chunkids = [x[0] for x in batch]
    docids = [x[1] for x in batch]
    docs = [quick_text_normalize(x[2]) for x in batch]
    messages = [
        [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": f'Document: "{doc}" Output: '},
        ] for doc in docs
    ]
    return chunkids, docids, messages


# llm.get_default_sampling_params()

# # len(llm.get_tokenizer().apply_chat_template(message))

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
        max_num_batched_tokens=512 * batch_size,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        generation_config="auto",
        max_model_len=10000,  # Limit context window
        max_num_seqs=batch_size,  # Limit batch size
        gpu_memory_utilization=0.95,
        disable_cascade_attn=True,  # Avoid gibberish output due to batch inference
        # model="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    )

    temporal_jsonl = []

    for chunkid, docid, messages in tqdm(loader):

        start_time = time.perf_counter()
        outputs = llm.chat(messages=messages,sampling_params=sampling_params)
        end_time = time.perf_counter()
        print(f"Time taken for inference: {end_time - start_time:.2f}")
        for _chunkid, _docid, output in zip(chunkid, docid, outputs):
            generated_text = output.outputs[0].text
            print(f"Generated text: {generated_text!r}")
            all_temp = generated_text.strip("\n").split(",")
            deduped_temp = list(set(all_temp))
            print("Number of temporal experessions extracted:", len(generated_text.split(", ")))
            print("Deduplicated temporal expressions:", len(deduped_temp))   

            print("-" * 50)
            temporal_jsonl.append(
                {
                    "chunkid": _chunkid,
                    "docid": _docid,
                    "temporal": ",".join(deduped_temp).strip(),
                }
            )
            # st()

    write_json(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/train/corpus_temporal_refined.jsonl",
        temporal_jsonl,
        jsonl=True,
    )

    del llm
    cleanup_dist_env_and_memory()
    #     st()
    #     message = [{"role": "user", "content": "why do you genearte 'after the Secretary of the Interior designation, since the establishment of the first federal area to protect wildlife'"}]


if __name__ == "__main__":
    main()
