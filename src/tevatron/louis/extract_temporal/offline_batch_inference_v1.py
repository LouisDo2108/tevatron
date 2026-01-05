import re
import os
import sys
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset, default_collate
from dataclasses import dataclass
from datasets import load_dataset
import msgspec
from pdb import set_trace as st
from tqdm.auto import tqdm
from collections import defaultdict
from pprint import pprint
from typing import Dict

from transformers import AutoTokenizer, HfArgumentParser, set_seed
from tevatron.retriever.arguments import DataArguments, ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.arguments import DataArguments

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from vllm.distributed import cleanup_dist_env_and_memory
from json_repair import repair_json

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

from pydantic import BaseModel
from typing import List
from enum import Enum

from sutime import SUTime

sutime = SUTime(mark_time_ranges=True, include_range=True)

# Allen relations
class AllenRelation(str, Enum):
    Before = "Before"
    After = "After"
    Meets = "Meets"
    MetBy = "MetBy"
    Overlaps = "Overlaps"
    OverlappedBy = "OverlappedBy"
    Starts = "Starts"
    StartedBy = "StartedBy"
    During = "During"
    Contains = "Contains"
    Finishes = "Finishes"
    FinishedBy = "FinishedBy"
    Equals = "Equals"
    Empty = "Empty"

# Temporal type
class TemporalQueryType(str, Enum):
    Explicit = "Explicit"
    Implicit = "Implicit"
    # Ordinal = "Ordinal"
    TemporalAnswer = "TemporalAnswer"

# # Reasoning level
# class ReasoningLevel(str, Enum):
#     L2 = "L2"  # Time-Event
#     L3 = "L3"  # Event-Event

# Passage model
class Passage(BaseModel):
    docid:int # The corresponding docid in the corpus
    text:str
    temporal:List[str]
    temporal_query_type:TemporalQueryType
    allen_relation:AllenRelation
    # reasoning_level:ReasoningLevel

# Full JSON schema
class TemporalAnnotation(BaseModel):
    query_id:int
    query:str
    temporal:List[str]
    positive_passages:List[Passage]
    negative_passages:List[Passage]


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



# Precompile regexes for speed if this runs many times
_whitespace_re = re.compile(r"\s+")
_punct_spacing_re = re.compile(r"\s*([,.;:?!])\s*")
_char_space_re = re.compile(r"\b(?:[A-Za-z]\s+)+[A-Za-z]\b")
# _sentence_re = re.compile(r"(?=\S)(?:[A-Z][a-z]{0,3}\.|[^.?!;:]|\.(?!\s+[A-Z]))*.?")

def normalize_and_split_string_by_punctuation(text):    
    # Normalize whitespace
    text = _whitespace_re.sub(" ", text.strip())

    # Fix spacing around punctuation (any of ,.;:?!)
    text = _punct_spacing_re.sub(r"\1 ", text)
    
    # Fix character-level split words
    text = _char_space_re.sub(lambda m: m.group(0).replace(" ", ""), text)    

    # punct_regex = r"(?=\S)(?:[A-Z][a-z]{0,3}\.|[^.?!;:]|\.(?!\s+[A-Z]))*.?"
    punct_regex = r"(?=\S)(?:[A-Z][a-z]{0,3}\.|[^.?!]|\.(?!\s+[A-Z]))*.?"
    return [x.strip() for x in re.findall(punct_regex, text)]


prompt_template = """You are a temporal annotation expert for information retrieval. We provide you with a query that may contain one or many temporal expressions and the corresponding SUTIME's output for your reference. Your job is to generate high-quality positive passages and negative passages for temporal contrastive learning.
Your final output are annotated JSONs (no indentation) that strictly follow the provided JSON schema.
Your temporal annotations must be more precise and reliable than SUTIME or HeidelTime, with top-tier accuracy. 
You must follow the definitions and instructions below.

* Allen relations and descriptions:
- Before: Did ‘Event A’ occur before ‘Event B’ without any overlap between the two events?
- After: Did ‘Event A’ occur after ‘Event B’ without any overlap between the two events?
- Meets: Did ‘Event A’ end in the same time as ‘Event B’ began? Answer True or False.
- MetBy: Did ‘Event B’ end in the same time as ‘Event A’ began? Answer True or False.
- Overlaps: Did ‘Event A’ begin before ‘Event B’ and end before ‘Event B’ ended, with some overlap between the two events?
- OverlappedBy: Did ‘Event B’ begin before ‘Event A’ and end before ‘Event A’ ended, with some overlap between the two events?
- Starts: ‘Event A’ begin in the same time as ‘Event B’, but end before ‘Event B’ ended?
- StartedBy: Did ‘Event B’ begin in the same time as ‘Event A’, but end before ‘Event A’ ended?
- During: Did ‘Event A’ begin after ‘Event B’ began and end before ‘Event B’ ended, being entirely contained within ‘Event B’?
- Contains: Did ‘Event A’ begin before ‘Event B’ began and end after ‘Event B’ ended, entirely containing ‘Event B’?
- Finishes: Did ‘Event A’ begin after ‘Event B’ began and end in the same time as ‘Event B’?
- FinishedBy: Did ‘Event B’ begin after ‘Event A’ began and end in the same time as ‘Event A’?
- Equals: Did ‘Event A’ begin in the same time as ‘Event B’ and end in the same time as ‘Event B’?
- Empty: a special case for TemporalAnswer where there is no temporal expression.

* Temporal signals (examples of what can appear in queries or passages): before, prior to, until, after, following, since, in, on, as of, for duration, during, while, when, from...to..., between, by, up to, first, last, around, as soon as, as long as, for, over, all through, throughout, etc.

* Taxonomy of "positive_passages" and "negative_passages" with examples that you should generate:
- Explicit temporal constraints (i.e., always have clear temporal expressions that can be anchored to a specific datetime): "who won the state of Texas in 2008?"; "what kind of government does Iran have after 1979?".
- Implicit temporal constraints (i.e., events that cannot be anchored to specific datetime. The rule of thumb is: if there is date or month or year in the question, like "before the **2004** general election", it is not implicit temporal): "who was the president after JFK died?"; "what team did Michael Jordan play for after the Bulls?".
- TemporalAnswer (i.e., Questions that inquire the datetime of an event instead of a general question with temporal constraints, usually starts with "when" or "what/which + day/date/month/year". There is no temporal expression in it.): "what year did the Knicks win the championship?"; "when was the United Nations founded?".

* Instructions:
1. You must output one or multiple valid JSONs, delimited by a newline, strictly following this pydantic schema:
{'$defs': {'AllenRelation': {'enum': ['Before', 'After', 'Meets', 'MetBy', 'Overlaps', 'OverlappedBy', 'Starts', 'StartedBy', 'During', 'Contains', 'Finishes', 'FinishedBy', 'Equals', 'Empty'], 'title': 'AllenRelation', 'type': 'string'}, 'Passage': {'properties': {'docid': {'title': 'Docid', 'type': 'integer'}, 'text': {'title': 'Text', 'type': 'string'}, 'temporal': {'items': {'type': 'string'}, 'title': 'Temporal', 'type': 'array'}, 'temporal_query_type': {'$ref': '#/$defs/TemporalQueryType'}, 'allen_relation': {'$ref': '#/$defs/AllenRelation'}}, 'required': ['docid', 'text', 'temporal', 'temporal_query_type', 'allen_relation'], 'title': 'Passage', 'type': 'object'}, 'TemporalQueryType': {'enum': ['Explicit', 'Implicit', 'TemporalAnswer'], 'title': 'TemporalQueryType', 'type': 'string'}}, 'properties': {'query_id': {'title': 'Query Id', 'type': 'integer'}, 'query': {'title': 'Query', 'type': 'string'}, 'temporal': {'items': {'type': 'string'}, 'title': 'Temporal', 'type': 'array'}, 'positive_passages': {'items': {'$ref': '#/$defs/Passage'}, 'title': 'Positive Passages', 'type': 'array'}, 'negative_passages': {'items': {'$ref': '#/$defs/Passage'}, 'title': 'Negative Passages', 'type': 'array'}}, 'required': ['query_id', 'query', 'temporal', 'positive_passages', 'negative_passages'], 'title': 'TemporalAnnotation', 'type': 'object'}.

2. "query_id": must be assigned with the provided "query_id".

3. “query":
- A query is a provided sentence that may contain one or many temporal expressions.
- Please note that SUTIME only provides explicit temporal expressions for the query and they are not perfect; therefore, you must double-check them and further detect additional explicit and implicit temporal expressions such as events.
- You must extract all events that are anchored to specific datetime (e.g., “2000 FA Cup Final”, “the 2007 election”, etc.).
- The "temporal" field should be extracted as written from the query's text and they must be concise, such as: "in April, 1906", "2 July 2010", "2004 general election", etc.

4. "positive_passages":
- Natural, QA-style questions with diverse phrasing that seek information from the query. Each question must contain exactly one extractable temporal expression, logically align with one of the query's temporal expressions, yet be diverse in Allen relations.
- Based on the query's temporal expressions, you must generate "positive_passages" that cover all "TemporalQueryType", including 'Explicit', 'Implicit', and 'TemporalAnswer', when possible:
    - You MUST prioritise generating questions with explicit temporal constraints, like "in 2010", "from 2010 to 2015", etc. They must logically align with the query’s temporal expression(s), such as being equals, overlapping with, or being contained within the query’s temporal expression(s). If the query has multiple temporal expressions, the passage must logically align with at least one of them.
    - You should also prioritise generating questions with implicit temporal constraints, such as "after EVENT", "before EVENT". These must logically align with the query’s temporal expression(s).
    - For both explicit and implicit "TemporalQueryType" questions, you must not use phrases like what/which date/day/month/year/time or when etc., that inquire about time. You must not confuse this with TemporalAnswer questions.
    - Finally, you can generate "TemporalAnswer" (asking for datetime/duration/time-range of an EVENT, e.g., "When did EVENT happen?", "What time did he arrive?"). For this type of question, you must not add any temporal expression and you must set: "TemporalQueryType": "TemporalAnswer", "allen_relation": "Empty", and "temporal": [].
- "temporal" field: must extract all exact text spans of the temporal expressions that exist in the generated question (not normalized or paraphrased). Regarding explicit and implicit "TemporalQueryType", they must be concise temporal expressions as written, such as "after July 2010", "from 2012 to 2014", that are not just normalized dates. For instance, instead of "In 1906 he moved with his family to a farm", prefer the concise "In 1906" and keep prepositions or context words that anchor the time, e.g., 'from', 'in', 'after', etc. Regarding "TemporalAnswer" passages, the "temporal" field must be an empty list.
- Allen relation: consider Passage = Event A and Query = Event B. Must be correct and align with the provided definition.
- “docid”: must remain the same as provided.
- Quantity: For each split/rewritten query, you must generate a list of 5 high-quality positive passages, prioritizing quality over quantity.

5. "negative_passages": Exactly the same as "positive_passages" in format, except that either have completely irrelevant contexts but with similar temporal expressions as the query, or have temporal expressions that are conflicted or do not exist in the query. Based on the query's temporal expressions, you must generate "negative_passages" that cover all "TemporalQueryType", including "Explicit", "Implicit", and "TemporalAnswer", when possible. They must be hard, temporally-confused, yet diverse in Allen relations questions that cover all following cases:
- Case 1: Questions with temporal constraints that DO NOT overlap with the query’s temporal expression(s): 
    - If the query is a span (e.g., 2005–2007), the temporal constraint must fall entirely outside (before 2005 or after 2007).
    - Open-ended intervals like “after 2010” or “before 2009” are valid only if they do not overlap with any of the query’s span.
    - Adjacent years or ranges (e.g., 2009 vs. query 2010).
    - Shifted intervals that look plausible but do not overlap (e.g., 2012–2014 for a query in 2010).
    - Misleading implicit cues that sound temporally close but are incorrect (e.g., “shortly after 2011” vs. query “in 2010”).
- Case 2: Questions with temporal constraints that overlap with the query’s temporal expression(s): Explicit/Implicit questions that have the same temporal expression as the query; however, seeking information for irrelevant event or entity. For example, the query is about Ronaldo's career in 2010, but the question is asking for Messi's career in 2010.
- Quantity: For each split/rewritten query, you must generate a list of 5 high-quality negative passages, prioritizing quality over quantity.
- Case 3: "TemporalAnswer" questions with no temporal expression or constraints. They are irrelevant to the query and can inquring for non-existent information from the query.

6. "temporal_query_type": Must be "Explicit", "Implicit", or "TemporalAnswer". If the temporal contains a clear date/month/year, it is "Explicit", NOT "Implicit". If the passage asks for which date/month/year of an event, it is "TemporalAnswer".
   
7. Final output: Only output valid JSON(s). Do not explain, add comments, or include extra text, since your output will be parsed automatically.

### Demonstration 1
Input:
docid: 2465
query_id: 0
query: "On 2 July 2010 , after helping Setúbal avoid top-flight relegation , Barbosa was released by Porto , signing a three-year contract with S.C . Braga."
SUTIME's output: [{'timex-value': '2010-07-02', 'start': 3, 'end': 14, 'text': '2 July 2010', 'type': 'DATE', 'value': '2010-07-02'}, {'timex-value': 'P3Y', 'start': 111, 'end': 121, 'text': 'three-year', 'type': 'DURATION', 'value': 'P3Y'}]

Output:
{"query_id":0,"query":"On 2 July 2010, after helping Setúbal avoid top-flight relegation, Barbosa was released by Porto, signing a three-year contract with S.C. Braga.","temporal":["2 July 2010","three-year contract"],"positive_passages":[{"docid":2465,"text":"Which team did Hélder Barbosa play for from 2010 to 2013?","temporal":["from 2010 to 2013"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for on 2 July 2010?","temporal":["2 July 2010"],"allen_relation":"Equals","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa join after leaving Porto in July 2010?","temporal":["after leaving Porto in July 2010"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa sign a three-year contract with?","temporal":["three-year contract"],"allen_relation":"Equals","temporal_query_type":"Explicit"},{"docid":2465,"text":"After being released by Porto, which team did Hélder Barbosa sign a contract with?","temporal":["After being released by Porto"],"allen_relation":"MetBy","temporal_query_type":"Implicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for between July 2010 and July 2013?","temporal":["between July 2010 and July 2013"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2465,"text":"When did Hélder Barbosa sign a contract with S.C. Braga.?","temporal":[],"allen_relation":"Empty","temporal_query_type":"TemporalAnswer"},{"docid":2465,"text":"How long did Hélder Barbosa's contract with S.C. Braga last?","temporal":[],"allen_relation":"Empty","temporal_query_type":"TemporalAnswer"}],"negative_passages":[{"docid":2465,"text":"Which team did Hélder Barbosa play for from 2002 to 2009?","temporal":["from 2002 to 2009"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for between 2006 and 2009?","temporal":["between 2006 and 2009"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for after 2014?","temporal":["after 2014"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for during 2008?","temporal":["during 2008"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for prior to 2010?","temporal":["prior to 2010"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Ronaldo play for in July 2010?","temporal":["in July 2010"],"allen_relation":"Equals","temporal_query_type":"Explicit"}]}
### End of Demonstration 1

### Demonstration 2
Input:
docid: 2466
query_id: 1
query: "Rarely used in the first months , he began gaining more playing time after the January 2011 departure of Matheus , who left for a team in Ukraine , and contributed four league goals in an eventual fourth-place finish ."
SUTIME's output: [{'timex-value': 'PXM', 'start': 15, 'end': 31, 'text': 'the first months', 'type': 'DURATION', 'value': 'PXM'}, {'timex-value': '2011-01', 'start': 78, 'end': 90, 'text': 'January 2011', 'type': 'DATE', 'value': '2011-01'}]

{"query_id":1,"query":"Rarely used in the first months, he began gaining more playing time after the January 2011 departure of Matheus , who left for a team in Ukraine , and contributed four league goals in an eventual fourth-place finish .","temporal":["after the January 2011 departure of Matheus"],"positive_passages":[{"docid":2466,"text":"Which team did Hélder Barbosa play for between January 2011 and December 2011?","temporal":["between January 2011 and December 2011"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2466,"text":"Which team did Hélder Barbosa play for as of January 2011?","temporal":["as of January 2011"],"allen_relation":"Equals","temporal_query_type":"Explicit"},{"docid":2466,"text":"Which team did Hélder Barbosa play for after Matheus departed in January 2011?","temporal":["after Matheus departed in January 2011"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2466,"text":"Which team did Hélder Barbosa contribute goals to during the 2011 season?","temporal":["during the 2011 season"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2466,"text":"Which team did Hélder Barbosa play for throughout 2011?","temporal":["throughout 2011"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2466,"text":"When did Hélder Barbosa start getting more playing time?","temporal":[],"allen_relation":"Empty","temporal_query_type":"TemporalAnswer"},{"docid":2466,"text":"When did Hélder Barbosa begin gaining more playing time?","temporal":[],"allen_relation":"Empty","temporal_query_type":"TemporalAnswer"}],"negative_passages":[{"docid":2466,"text":"Which team did Hélder Barbosa play for before 2010?","temporal":["before 2010"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2466,"text":"Which team did Hélder Barbosa play for after 2013?","temporal":["after 2013"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2466,"text":"Which team did Hélder Barbosa play for in 2007?","temporal":["in 2007"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2466,"text":"Which team did Hélder Barbosa play for prior to joining Braga?","temporal":["prior to joining Braga"],"allen_relation":"Before","temporal_query_type":"Implicit"},{"docid":2466,"text":"Which team did Hélder Barbosa play for during 2014?","temporal":["during 2014"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2466,"text":"After the January 2011 departure of Matheus, did Ronaldo get more playing time?","temporal":["After the January 2011 departure of Matheus"],"allen_relation":"Equals","temporal_query_type":"Explicit"}]}
### End of Demonstration 2"""

# Note how one long "query" input is split into non-overlapping smaller queries with ideally one temporal expression each to avoid ambiguity in temporal information.

temporal_answer_list = [
    # Point in time
    "what day",
    "what time",
    "what date",
    "what month",
    "what year",
    "which day",
    "which time",
    "which date",
    "which month",
    "which year",
    "when did",
    "when was",
    "when were",
    "at what time",
    "at what date",
    "at what year",
    "on what day",
    "on what date",
    "on what year",

    # Duration / span
    "how long",
    "how many days",
    "how many weeks",
    "how many months",
    "how many years",
    "for how long",
    "over what period",
    "during what years",
    "during which year",
    "for what duration",
    "in what year range",
    "between what years",

    # Frequency / recurrence
    "how often",
    "how frequently",

    # Relative temporals
    "since when",
    "until when",
    "from when",
    "from what year",
    "from what date",
    "to what year",
    "to what date",
    "up to when",
    "as of when",
    "by what year",
    "by when",
    "around when",
    "at what age",
]


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
        
        docid = content["positive_passages"][0]["docid"]
        query = normalize_and_split_string_by_punctuation(content["query"])
        query_id = content["query_id"]

        return docid, query_id, query


def llm_collate(batch):
    """
    Collate function for encoding.
    :param features:list of (id, text, image) tuples
    but in this case, it's just image is None
    """
    
    messages = []
    
    for docid, query_id, queries in batch:
        for q in queries:
            """
            Edge case:
            sutime.parse("Which party did Samaras join the 2004 european elections?") -> Detect 2004
            sutime.parse("Which party did Samaras join the 2004 European elections?") -> Detect
            # sutime is case-sensitive
            """
            sutime_output = sutime.parse(q.lower()) 
            
            for t in sutime_output:
                s, e = t["start"], t["end"]
                t["value"] = q[s:e]
            
            if len(sutime_output) > 0:
                messages.append([
                    {"role":"system", "content":prompt_template},
                    {"role":"user", "content":f"Input:\ndocid: {docid}\nquery_id: {query_id}\nquery: {q}\nSUTIME's output: {sutime_output}\nOutput:"},
                ])

    return messages


def fix_implicit_temporal(passage, sutime):
    """
    Adjust temporal_query_type if 'implicit' but tagger detects explicit datetime.
    """
    if passage == {} or passage["temporal_query_type"] != TemporalQueryType.Implicit:
        return passage
    
    # Must have exactly 1 temporal
    if len(passage["temporal"]) != 1:
        return {}
    
    if len(sutime.parse(passage["temporal"][0].lower())) > 0:
        passage["temporal_query_type"] = TemporalQueryType.Explicit
        
    lower_passage = passage["text"].lower()
    
    # Make sure that the
    implicit_temporal = []
    for t in passage["temporal"]:
        start = lower_passage.find(t.lower())
        if start != -1:
            implicit_temporal.append(passage["text"][start:start+len(t)])    
    
    if not implicit_temporal:
        return {}
    
    passage["temporal"] = implicit_temporal
    return passage


def fix_temporal_answer(passage, sutime):
    if passage == {} or passage["temporal_query_type"] != TemporalQueryType.TemporalAnswer:
        return passage

    lower_passage = passage['text'].lower()
    if any(word in lower_passage for word in temporal_answer_list):
        for word in temporal_answer_list:
            if word in lower_passage:
                lower_passage = lower_passage.replace(word, "")
        
    # Ensure that it passes the SUTIME tagger.
    if len(sutime.parse(lower_passage)) > 0:
        return {}
    else:
        passage['temporal'] = []
        passage['allen_relation'] = AllenRelation.Empty
    # print(passage)
    return passage


def fix_explicit_temporal(passage, sutime):
    if passage == {} or passage["temporal_query_type"] != TemporalQueryType.Explicit:
        return passage
    
    # Must have exactly 1 temporal
    if len(passage["temporal"]) != 1:
        return {}
    
    lower_passage = passage["text"].lower()
    temporal = []

    for t in passage["temporal"]:
        start = lower_passage.find(t.lower())
        if start != -1:
            temporal.append(passage["text"][start:start+len(t)])

    if not temporal:
        return {}

    passage["temporal"] = temporal
    return passage


def validate_passages(passages, sutime):
    """
    Validate and filter passages.
    """
    valid = []
    
    for p in passages:
        p = fix_implicit_temporal(p, sutime)
        p = fix_explicit_temporal(p, sutime)
        p = fix_temporal_answer(p, sutime)
        
        if p:
            valid.append(p)
    return valid


def validate_temporal(js, max_temporal_expressions=1):
    lower_query = js["query"].lower()
    temporal = []
    
    # Ensure that the temporal is extracted as written from the original query
    for t in js["temporal"]:
        start = lower_query.find(t.lower())
        if start != -1:
            temporal.append(js["query"][start:start+len(t)])

    if len(temporal) == 0 or len(temporal) > max_temporal_expressions:
        # print(temporal)
        return []
    return temporal


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

    batch_size = training_args.per_device_train_batch_size

    dataset = LLMDataset(data_args)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=llm_collate,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    temporal_jsonl = []
    guided_decoding_params = GuidedDecodingParams(
        json=TemporalAnnotation.model_json_schema(),
    )
    
    sampling_params = SamplingParams(
        max_tokens=32768, 
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        guided_decoding=guided_decoding_params
    )
    
    llm = LLM(
        model="Qwen/Qwen3-4B-Instruct-2507",
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        generation_config="auto",
        max_model_len=32768,  # Limit context window
        max_num_seqs=batch_size,  # Limit batch size
        gpu_memory_utilization=0.95,
        disable_cascade_attn=True,  # Avoid gibberish output due to batch inference
        seed=42,
    )

    temporal_jsonl = []
    counter = 0

    for ix, i in enumerate(tqdm(loader)):
        
        # if ix == 1:
        #     break

        output = llm.chat(
            messages=i, 
            sampling_params=sampling_params,
            # chat_template_kwargs={"enable_thinking": True},
        )
        
        temp = []
        for sample in output:
            sample = sample.outputs[0].text # Get the generated text
            list_of_raw_js = []
            query_set = set()
            
            for x in sample.split("\n"): # Split by newline to get multiple JSONs if any
                parts = x.split(',{"query_id"')
                if len(parts) > 0:
                    json_strings = [parts[0]] + [',{"query_id"' + p for p in parts[1:]]
                    list_of_raw_js.extend(json_strings)
                else:
                    list_of_raw_js.append(x)
            
            for raw_js in list_of_raw_js:
                raw_js = raw_js.strip()
                if raw_js == "":
                    continue
                try:
                    js = TemporalAnnotation.model_validate_json(repair_json(raw_js, ensure_ascii=False)).model_dump()
                    
                    if len(js["temporal"]) == 0 or len(js["positive_passages"]) == 0 or len(js["negative_passages"]) == 0 or js["query"] in query_set:
                        continue
                    else:
                        # print(js["query"])
                        js["temporal"] = validate_temporal(js)
                        
                        if len(js["temporal"]) == 0:
                            continue
                        
                        # Validate passages
                        js["positive_passages"] = validate_passages(
                            js["positive_passages"], sutime
                        )
                        
                        if len(js["positive_passages"]) == 0:
                            continue
                        
                        js["negative_passages"] = validate_passages(
                            js["negative_passages"], sutime
                        )
                        
                        if len(js["negative_passages"]) == 0:
                            continue
                        
                        temp.append(js)
                        query_set.add(js["query"])
                        # positive = defaultdict(int)
                        # negative = defaultdict(int)
                        # allen_relation = defaultdict(int)

                        # for j in js["positive_passages"]:
                        #     positive[j["temporal_query_type"]] += 1
                        #     allen_relation[j["allen_relation"]] += 1
                        # for j in js["negative_passages"]:
                        #     negative[j["temporal_query_type"]] += 1
                        #     allen_relation[j["allen_relation"]] += 1
                        
                        # print(positive)
                        # pprint(positive)
                        # print(positive)
                        # pprint(negative)

                except Exception as e:
                    print("Error:", e)
                    print("Offending JSON:", raw_js)
                    continue
        temporal_jsonl.extend(temp)

        if (ix+1) % 10 == 0:
            write_json(f"/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/temporal_nobel_prize/enhanced_temporal/v3/{counter}.jsonl",temporal_jsonl,jsonl=True)
            temporal_jsonl = []
            counter += 1

    write_json(f"/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/temporal_nobel_prize/enhanced_temporal/v3/{counter}.jsonl",temporal_jsonl,jsonl=True)
    
    del llm
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
