# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai import OpenAI

from transformers import AutoTokenizer

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

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
    demo += f'Document: {doc} Output: {output}'
input_template="Input:  Document: {}. Output: {}"

# Temporal Tagger: SUTIME, Heideltime

# Define input template
# input_template = 'Previous output: "<PREVIOUS OUTPUT STARTS FROM HERE>" Output: <YOUR OUTPUT STARTS FROM HERE>'

# Assemble final prompt template
# Inspired by hipporag
prompt_template = f"""
You are an expert in preprocessing data. {task_description}
The input template is as follows: {input_template}. 
Some examples for your reference:
{demo.strip()}
Now, you should output the temporal expressions in the given documents.
"""

# Sample prompts.
prompts = [
    {"role": "system", "content": prompt_template},
    {
        "role": "user",
        "content": "Document: What was the working location for Alessandro Magnasco from 1735 to 1749? Output:",
    },
    {
        "role": "user",
        "content": "Document: What was the working location for Alessandro Magnasco in late 1700s? Output:",
    },
    {
        "role": "user",
        "content": "Document: What was the noble title of Arnulf of Carinthia from 880 to 887? Output:",
    },
    {
        "role": "user",
        "content": "Document: What was the working location for Alessandro Magnasco after 1755? Output:",
    },
]
# # Create a sampling params object.
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    response = client.chat.completions.create(model=model, messages=prompts)
    print(response.choices[0].message.content)

    from pdb import set_trace as st
    st()


if __name__ == "__main__":
    main()
