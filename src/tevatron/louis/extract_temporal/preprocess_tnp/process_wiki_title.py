from multiprocessing import Pool, cpu_count
from wikimapper import WikiMapper
import pandas as pd

import msgspec
from tqdm import tqdm
from tevatron.louis.src.utils import read_json, write_json

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

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