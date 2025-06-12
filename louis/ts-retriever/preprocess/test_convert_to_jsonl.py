from pathlib import Path
from pdb import set_trace as st
from pprint import pprint

import msgspec
import numpy as np
import pandas as pd

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

def convert_corpus_json_to_jsonl(input_file: str, output_file: str):
    with open(input_file, 'rb') as infile, open(output_file, 'wb') as outfile:
        input_json = decoder.decode(infile.read())

        output_jsonl = []
        for ix, v in enumerate(input_json):
            output_jsonl.append(
                {
                    "docid": ix,
                    "text": v,
                }
            )
        outfile.write(encoder.encode_lines(output_jsonl))


def convert_queries_json_to_queries_qrel_jsonl(input_file: str, output_file: str):

    with open(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/test/corpus.jsonl",
        "rb",
    ) as infile:
        doc_to_docid_dict = decoder.decode_lines(infile.read())
    docid_to_doc_dict = {v["docid"]: v["docid"] for v in doc_to_docid_dict}

    output_jsonl = []
    with open(input_file, 'rb') as infile:
        input_json = decoder.decode(infile.read())

    with open("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/test/qrel.txt", 'w') as outfile:
        for ix, v in enumerate(input_json):
            query = v["query"]
            positive_text = v["positive_text"]
            answer_names = [item.split('\n')[0] for item in positive_text]

            positive_passages = []
            for doc in positive_text:
                positive_passages.append({
                    "docid": docid_to_doc_dict[doc],
                    "text": doc,
                })
                outfile.write(f"{ix} 0 {docid_to_doc_dict[doc]} {1}\n")
                
            output_jsonl.append({
                    "query_id": ix,
                    "query": query,
                    "answers": answer_names,
                    "positive_passages": positive_passages,
                    # "negative_passages": [],
            })
    with open("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/test/query.jsonl", 'wb') as outfile:
        outfile.write(encoder.encode_lines(output_jsonl))


if __name__ == "__main__":
    # input_file = "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/test/corpus.json"
    # output_file = "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/test/corpus.jsonl"
    # convert_corpus_json_to_jsonl(input_file, output_file)

    input_file = "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/test/query.json"
    output_file = "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/test/query.jsonl"
    convert_queries_json_to_queries_qrel_jsonl(input_file, output_file)
    print(f"Converted {input_file} to {output_file}")
