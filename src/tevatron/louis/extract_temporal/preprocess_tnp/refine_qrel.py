import msgspec
import pandas as pd
from tqdm.auto import tqdm

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

from torch.utils.data import DataLoader, Dataset
from tevatron.louis.src.utils import read_json, write_json
import os

# export VLLM_USE_V1=1
# export TOKENIZERS_PARALLELISM=0
os.environ["VLLM_USE_V1"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "0"


from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

class TemporalQrelDataset(Dataset):
    """
    Dataset for temporal qrel annotation.
    Each item is a dict with 'system' and 'user' messages.
    """
    def __init__(self, df, prompt_template):
        """
        df: DataFrame with columns 'question', 'targets', 'text'
        prompt_template: string, system prompt
        """
        self.df = df
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['question']
        # targets can be a single string or list
        targets = row['targets']
        if isinstance(targets, list):
            answers = ", ".join([str(x) for x in targets])
        else:
            answers = str(targets)

        passage = row['text']

        index = idx

        user_content = f'Input: (Index: {index}, Question: "{question}", Answers: "{answers}", Passage: "{passage}")\nOutput:'

        return self.prompt_template, user_content

# --------------------------------
# Collate function for batching
# --------------------------------
def collate_fn(batch):
    """
    Converts a list of dicts into a batch suitable for LLM input.
    Returns a dict with:
        - 'system_messages': list of system messages
        - 'user_messages': list of user messages
    """

    return [
        [{"role":"system", "content":x[0]}, {"role":"user", "content":x[1]}]
        for x in batch
    ]


prompt_template = """You are a temporal relevance annotator for information retrieval.

Input format:
(index, query, list of answer_terms, passage)

Your task is to decide whether the passage is relevant to the query for qrel construction.

A passage is relevant only if BOTH conditions are true:
1. The passage must have at least one term from answer_terms.
2. According to the answer term, the passage must satisfy all temporal constraints expressed or implied in the query (e.g., specific year, time range, in/before/after/from...to... relations).
3. Mentions of the answer terms outside the correct time window are NOT relevant.

Output format:
- You must output the integer index only if the passage is relevant.
- Otherwise, you must output only "-1".
- You must not output explanations or additional text.

### Demonstration 1
Input:
(Index: 1, Question: Who was the head coach of the team 1. FC Köln from Nov 2019 to Nov 2020?, Answers: ['Markus Gisdol'], Passage: "'Title: 1._FC_Köln. Section: Decline and changes (2018–). FC Saarbrücken, the club decided to terminate Beierlorzer's contract on 9 November 2019. Sporting director Armin Veh, who weeks earlier had announced that he would not extend his contract with the club, was also dismissed from his position. On 18 November, former HSV manager Markus Gisdol was appointed to the club's head coaching position, while Horst Heldt was made sporting director.  Both signed contracts until 2021. After avoiding relegation at the end of the season, Gisdol's contract was extended until 2023. )"
Output: -1
### End of Demonstration 1

### Demonstration 2
Input:
(Index: 2, Question: Who was the head coach of the team 1. FC Köln from Nov 2019 to Nov 2020?, Answers: ['Markus Gisdol'], Passage: "Title: 1._FC_Köln. Section: Decline and changes (2018–).   The club found itself in a renewed relegation during the 2020–21 season. On 11 April 2021, after losing to relegation rival Mainz 05, Gisdol was dismissed from his position as head coach. The next day, it was announced that Friedhelm Funkel would take over head coaching duties until the end of the season. On 11 May, it was reported that SC Paderborn manager Steffen Baumgart would succeed Funkel as head coach at the beginning of the 2021–22 season. )"
Output: -1
### End of Demonstration 2

### Demonstration 3
Input:
(Index: 3, Question: Who was the head coach of 1. FC Köln from Nov 2019 to Nov 2020?, Answers: ['Markus Gisdol'], Passage: "'Title: 1._FC_Köln. Gisdol was appointed head coach in November 2018 and left in October 2019 before the season ended.'")
Output: -1
### End of Demonstration 3
"""

if __name__ == "__main__":

    sampling_params = SamplingParams(
        max_tokens=16, 
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0
    )

    llm = LLM(
        model="Qwen/Qwen3-4B-Instruct-2507",
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        generation_config="auto",
        max_model_len=32768,  # Limit context window
        max_num_seqs=4,  # Limit batch size
        gpu_memory_utilization=0.95,
        disable_cascade_attn=True,  # Avoid gibberish output due to batch inference
        seed=42,
    )
    
    df = pd.read_parquet("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/test/corpus23009_sentencechunk128_with_test_query.parquet")
    
    dataset = TemporalQrelDataset(df, prompt_template)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)

    result = []

    for batch in tqdm(dataloader):
        output = llm.chat(
            messages=batch, 
            sampling_params=sampling_params,
            # chat_template_kwargs={"enable_thinking": True},
        )
        for out in output:
            out = out.outputs[0].text
            if int(out) != -1:
                result.append(int(out))

        