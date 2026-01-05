from multiprocessing import Pool, cpu_count

from wikimapper import WikiMapper

from pdb import set_trace as st
from pprint import pprint
from collections import defaultdict
import pandas as pd

import msgspec
from tqdm import tqdm
from pathlib import Path

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

def read_json(file_path, jsonl=False):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError("filepath is not a file")
    # if not file_path.suffix == ".jsonl" and jsonl:
    #     raise ValueError("the file is not jsonl")
    # if not file_path.suffix == ".json" and not jsonl:
    #     raise ValueError("the file is not json")
    file_path = file_path.__str__()
    
    output = []

    with open(file_path, "rb") as file:
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


def read_tsv(file_path, row_names=None):
    file_path = Path(file_path)
    if file_path.is_file() and file_path.suffix == ".tsv" :
        temp = pd.read_csv(file_path, sep='\t', names=row_names)
    else:
        raise ValueError("filepath is not a file or it is not a tsv file.")
    return temp


def process_title(x):
    # extract, normalize, and map title to ID
    return mapper.title_to_id("_".join(x['title'][2:-2].split(" ")))

if __name__ == '__main__':
    
    mapper = WikiMapper("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/flashrag/index_enwiki-20220820.db")
    
    for i in range(1, 11):
        temp_jsonl = []
        with open(f"/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/flashrag/split/{i}.jsonl") as f:
            temp_jsonl = []
            for ix, line in enumerate(tqdm(f)):
                temp_jsonl.append(decoder.decode(line))
            # temp_jsonl = [decoder.decode(line) for line in tqdm(f)]
            
        temp_jsonl_df = pd.DataFrame(temp_jsonl)
        
        with Pool(63) as pool:  # use all available cores
            results = list(tqdm(
                pool.imap(process_title, temp_jsonl), 
                total=len(temp_jsonl)
            ))
            
        temp_jsonl_df['id'] = results
        temp_jsonl_df.to_parquet(f"/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/flashrag/split/{i}.parquet")