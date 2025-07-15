# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai import OpenAI

from transformers import AutoTokenizer

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

task_description = 'Task Description: Extract all temporal expressions that convey time-related information in the text. Output all temporal expressions as a comma-separated list, do not include any note. These include explicit dates (e.g., "March 2023"), implicit cues (e.g., "recently", "during his presidency", "after a major power outage"), relative expressions (e.g., "last week"), event-based references (e.g., "2024 Olympics", "2009 Georgian Womens Championship"), and durations or time spans (e.g., "from 2013 to 2014").   If the text belongs to diachronic corpora (documents spanning long time periods), perform timeline summarization like a professional historian or journalist. If it belongs to synchronous corpora (snapshots of a specific point in time), extract all relevant temporal expressions for that context. Normalize temporal expressions where possible into consistent formats.'
demo = "Document: The history of Poland from 1945 to 1989 spans the period of Marxist–Leninist regime in Poland after the end of World War II. These years, while featuring general industrialization, urbanization and many improvements in the standard of living,[a1] were marred by early Stalinist repressions, social unrest, political strife and severe economic difficulties. Output: 1945 to 1989, Marxist–Leninist regime in Poland, after the end of World War II. Document: He signed a two-year contract with League One side Crewe Alexandra in June 2013 after manager Steve Davis paid Macclesfield an undisclosed fee. However, he played just five games for the Railwaymen, being sent on two loan spells to Lincoln City, before returning to Macclesfield Town in February 2015. He then played for newly relegated Notts County in League Two for two seasons. In July 2017, Audel joined Barrow, moving to Welling United a year later.Output: two-year contract, June 2013, February 2015, two seasons, July 2017, a year later. Document: A week later Rangers made an improved offer of £1m , which was also rejected . Hearts then placed a £3m price tag on Wallace . Wolverhampton Wanderers , then a Premier League side , had also expressed an interest in Wallace but did not make any offers . On 21 July 2011 , a bid of £1.5m from Rangers was accepted by Hearts. Output: a week later, 21 July 2011. Document: We design ablation experiments for the proposed linking method, graph construction method, and triple filtering method, with the results reported in Table 4. Each introduced mechanism boosts HippoRAG 2. Output: Document: He won his first league title in 2005 and was the clubs first-choice goalkeeper up until 2011 , making over 200 appearances for América . That summer Ochoa was transferred to Ajaccio in France . He spent three seasons with the club until their relegation from Ligue 1 . In 2014 , Ochoa joined Málaga but failed to establish himself in the team . In July 2016 , he joined Granada on a season-long loan . In July 2017 , he joined Standard Liège. Output: 2005, until 2011, until relegation from Ligue 1, that summer, three seasons, 2014, July 2016, July 2017. Document: What happened in Poland after World War II and before 1960?. Output: after World War II, before 1960, after 1945, from 1946.\
"
input_template="Input:  Document: {}. Output: {}"

# Temporal Tagger: SUTIME, Heideltime 

hipporag_prompt_template = f"\
You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide.\
Your task is to {task_description}.\
Some examples for your reference: f{demo}\
You must extract relevant temporal information that is strongly connected to the content of the document. The goal is to aid temporal reasoning and improve the accuracy of answers. Only use time expressions explicitly mentioned or clearly implied in the document — do not invent or hallucinate temporal information. In specific cases where historical facts are universally known and confidently implied (e.g., 'after 1945' or 'from 1946' for events following World War II), you may include them. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. The future of critical decisionmaking relies on your ability to accurately filter and present relevant temporal information.\
{input_template}\
"


# Sample prompts.
prompts = [
    {"role": "system", "content": hipporag_prompt_template},
    {
        "role": "user",
        "content": "Document: 2010 . Paikidze jointly won the womens open event of the Moscow Open , won the Moscow Womens Championship and the Russian Womens Championship qualifier ( Higher League ) , and finished fourth in the Russian Womens Championship Superfinal . In the same year , she was awarded the title of Woman Grandmaster ( WGM ) , for her results in the 2008 Aeroflot Open , 2009 Georgian Womens Championship , where she came equal second , and the 2010 Moscow Open. Output: ",
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
    print("Document:",prompts[1]["content"][10:-8])
    print("#########")
    print("Temporal information:", response.choices[0].message.content)

    from pdb import set_trace as st
    st()


if __name__ == "__main__":
    main()
