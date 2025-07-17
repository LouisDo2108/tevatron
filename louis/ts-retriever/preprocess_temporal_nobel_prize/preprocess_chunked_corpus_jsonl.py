# %%
import msgspec
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
import re 
from sutime import SUTime

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

global sutime, heideltime_parser
sutime = SUTime(mark_time_ranges=True, include_range=True)

from python_heideltime import Heideltime

heideltime_parser = Heideltime()
heideltime_parser.set_document_type("NEWS")

from concurrent.futures import ThreadPoolExecutor
encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

timex_text_pattern = re.compile(r"<TIMEX3[^>]*>(.*?)</TIMEX3>")
timex_value_pattern = re.compile(r'<TIMEX3[^>]*\bvalue="([^"]+)"')
timex_pattern = re.compile(r'<TIMEX3[^>]*\bvalue="([^"]+)"[^>]*>(.*?)</TIMEX3>')

# %% [markdown]
# # Utils

# %%
def read_json(file_path, jsonl=False):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError("filepath is not a file")
    # if not file_path.suffix == ".jsonl" and jsonl:
    #     raise ValueError("the file is not jsonl")
    # if not file_path.suffix == ".json" and not jsonl:
    #     raise ValueError("the file is not json")
    file_path = file_path.__str__()

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


def parse_heideltime_result(heideltime_result, flatten=False):
    # root = etree.fromstring(heideltime_result, parser=parser)

    # values = []
    # for el in root.iter():
    #     if el.tag == "TIMEX3":
    #         value = el.attrib.get("value", "")
    #         if value and "2025" not in value and not value.startswith("P"):
    #             values.append({"text": el.text or "", "value": value})

    heideltime_result = heideltime_result.split("\n")[3]
    values = timex_text_pattern.findall(heideltime_result)

    # if flatten:
    #     return [field for item in values for field in (item["text"], item["value"])]

    return values


def parse_sutime_result(sutime_result, flatten=False):
    values = [x.get("text") for x in sutime_result if x.get("text")]
    # for x in sutime_result:
    #     text = x.get("text", "")
    #     # value = x.get("value", "")

    #     if isinstance(value, dict):
    #         value = ""
    #     elif not value or ("2025" not in value and value.startswith("P")):
    #         value = ""

    #     if flatten:
    #         values.append(text)
    #         if value:
    #             values.append(value)
    #     else:
    #         values.append({"text": text, "value": value})

    return values

# %% [markdown]
# # Read the corpus data

# %%
DATA_PATH = Path("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/temporal_nobel_prize")
SPLIT = "train"

CORPUS_TEMPORAL_JSONL_PATH = DATA_PATH / SPLIT / "chunked/corpus_temporal_refined.jsonl"
corpus_temporal_jsonl = read_json(CORPUS_TEMPORAL_JSONL_PATH, jsonl=True)

CHUNKED_CORPUS_JSONL_PATH = DATA_PATH / SPLIT / "chunked/corpus.jsonl"
chunked_corpus_jsonl = read_json(CHUNKED_CORPUS_JSONL_PATH, jsonl=True)

STITCHED_CORPUS_JSONL = []
for x, y in zip(chunked_corpus_jsonl, corpus_temporal_jsonl):
    x.update(y)
    STITCHED_CORPUS_JSONL.append(x)


def init_worker():
    global sutime, heideltime_parser
    
    from sutime import SUTime
    sutime = SUTime(mark_time_ranges=True, include_range=True)

    from python_heideltime import Heideltime
    heideltime_parser = Heideltime()
    heideltime_parser.set_document_type("NEWS")


def get_temporal_set(item):

    text = item["text"]
    
    if item["temporal"] == "":
        return item
    
    llm_result= item["temporal"].split(",")
    sutime_result = parse_sutime_result(sutime.parse(text))
    heideltime_result = parse_heideltime_result(heideltime_parser.parse(text))

    item["temporal"] = set(sutime_result).union(heideltime_result).union(llm_result)
    return item


with ThreadPoolExecutor(max_workers=24, initializer=init_worker) as executor:
    futures = [executor.submit(get_temporal_set, item) for item in STITCHED_CORPUS_JSONL]
    for _ in tqdm(as_completed(futures), total=len(futures)):
        pass  # We just wait for all tasks to complete, since results are stored in-place

write_json(
    "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/temporal_nobel_prize/train/chunked/temporal_sutime_heideltime.jsonl",
    STITCHED_CORPUS_JSONL,
    jsonl=True,
)
