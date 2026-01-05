prompt_template = """
You are a temporal annotation expert for information retrieval. We provide you with a previously annotated JSON, consisting of a query with its previously annotated positive and negative passages. Your job is to output better fully annotated JSONs (no indentation) for temporal contrastive learning. We also provide you with SUTIME tagger's output of the query for your reference. Your annotations must be more precise and reliable than SUTIME or HeidelTime, with top-tier accuracy. Follow the definitions and rules below.

Allen relations:
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
- Empty (Special case for TemporalAnswer type where there is no temporal expression in the passage)

Temporal signals (examples of what can appear in queries or passages): before, prior to, until, after, following, since, in, on, as of, for duration, during, while, when, from...to..., between, by, up to, first, last, around, as soon as, as long as, for, over, all through, throughout, etc.

Different type of temporal questions with examples that you should generate:
- Explicit temporal: "who won the state of texas in 2008?"; "what kind of government does iran have after 1979?" (i.e., always have a clear temporal expression that can be anchored to a specific datetime).
- Implicit temporal: "who was the president after jfk died?"; "what team did michael jordan play for after the bulls?" (i.e., events that cannot be anchored to specific datetime. The rule of thumb is: if there is date or month or year in the question, like "before the **2004** general election", it is not implicit temporal).
- Temporal answer: "what years did the knicks win the championship?"; "when was the united nations founded?" (i.e., Inquiring the datetime of an event instead of a general question with temporal constraint, usually starts with "when" or "what date/month/year").


Rules:
1. Query splitting (Critical):
    - Purpose: reduce temporal ambiguity by breaking long queries into smaller, unambiguous ones.
    - Process (scan left to right, sentence by sentence):
        - If a sentence has single temporal expressions: start a split, then append following sentences without temporal expressions until the next sentence containing one.
        - If one of the reference SUTIME's output belongs to this sentence, you should definitely consider splitting it.
        - Prioritize splits with explicit datetimes or events anchored to time (e.g., “2000 FA Cup Final”, “the 2007 election”).
        - Never repeat current split in subsequent splits.
    - All "temporal" of each query split should be extracted, but they must be concise, such as: "in April, 1906", "2 July 2010", "2004 general election", etc., that summarize the temporal context.
    - Splits must be mutually exclusive (no overlapping text).
    - If the query has exactly one temporal expression, keep it as is.
    - Examples: 
        - "In August 2005, Brown joined Uruguayan giants Peñarol and in August 2006 he made his debut for fellow Uruguayans Tacuarembó." can be split into two queries:
            1) "In August 2005, Brown joined Uruguayan giants Peñarol"
            2) "in August 2006 he made his debut for fellow Uruguayans Tacuarembó."
        - "He returned to LASK for the 2009–10 season and was made captain. In Summer 2010 he moved to fellow top flight side Austria Vienna, where he signed a contract until 2013. Here, he played European football with the team in the UEFA Europa League." can be split into two queries:
            1) "He returned to LASK for the 2009–10 season and was made captain."
            2) "In Summer 2010 he moved to fellow top flight side Austria Vienna, where he signed a contract until 2013. Here, he played European football with the team in the UEFA Europa League."

2. You must output one or multiple valid JSONs, delimited by a newline, strictly following this pydantic schema:
{'$defs': {'AllenRelation': {'enum': ['Before', 'After', 'Meets', 'MetBy', 'Overlaps', 'OverlappedBy', 'Starts', 'StartedBy', 'During', 'Contains', 'Finishes', 'FinishedBy', 'Equals', 'Empty'], 'title': 'AllenRelation', 'type': 'string'}, 'Passage': {'properties': {'docid': {'title': 'Docid', 'type': 'integer'}, 'text': {'title': 'Text', 'type': 'string'}, 'temporal': {'items': {'type': 'string'}, 'title': 'Temporal', 'type': 'array'}, 'temporal_query_type': {'$ref': '#/$defs/TemporalQueryType'}, 'allen_relation': {'$ref': '#/$defs/AllenRelation'}}, 'required': ['docid', 'text', 'temporal', 'temporal_query_type', 'allen_relation'], 'title': 'Passage', 'type': 'object'}, 'TemporalQueryType': {'enum': ['Explicit', 'Implicit', 'TemporalAnswer'], 'title': 'TemporalQueryType', 'type': 'string'}}, 'properties': {'query_id': {'title': 'Query Id', 'type': 'integer'}, 'query': {'title': 'Query', 'type': 'string'}, 'temporal': {'items': {'type': 'string'}, 'title': 'Temporal', 'type': 'array'}, 'positive_passages': {'items': {'$ref': '#/$defs/Passage'}, 'title': 'Positive Passages', 'type': 'array'}, 'negative_passages': {'items': {'$ref': '#/$defs/Passage'}, 'title': 'Negative Passages', 'type': 'array'}}, 'required': ['query_id', 'query', 'temporal', 'positive_passages', 'negative_passages'], 'title': 'TemporalAnnotation', 'type': 'object'}.

3. Positive passages:
- Docid: must remain the same as provided.
- Temporal alignment:
    - A positive passage must overlap with or be contained within the query’s temporal anchor(s).
    - If the query has multiple anchors (e.g., 2005–2007), the passage must align with at least one valid span.
Question style:
- Must be natural, QA-style questions.
- Each question must contain exactly one extractable temporal expression.
- Use diverse temporal types:
    - Explicit (e.g., "in 2010"),
    - Implicit (e.g., "after EVENT", "before EVENT"),
    - TemporalAnswer (asking for a time directly, e.g., "When did EVENT happen?").
- Should base solely on the current query split, you must not assume any connection with other splits.
- Allen relation: must be accurately determined. Consider the generated passage’s temporal as event A and the query's temporal as event B.
- If the passage is of type TemporalAnswer, then: "allen_relation": "Empty" and "temporal": []
- Reuse: previously provided positives should be reused where possible, but ensure diversity (e.g., paraphrasing, different temporal types).
- Quantity: at least 5 high-quality positives per query, prioritizing quality over quantity.

4. Negative passages
- Docid: must remain the same as provided.
- Temporal conflict:
    - Negative passages must not overlap with any of the query’s temporal anchors.
    - If the query's temporal is 2005–2007, then a negative must fall completely before 2005 or after 2007 (e.g., 2004, 2008–2009).
    - Open-ended intervals like "after 2010" or "before 2009" are only valid if they do not overlap with the query’s span.
- Question style:
    - Must be natural, QA-style questions.
    - Each question must contain exactly one extractable temporal expression.
    - Only explicit or implicit temporal types are allowed (no TemporalAnswer).
- Should base solely on the current query split, you must not assume any connection with other splits.
- Allen relation: Allen relation: must be accurately determined. Consider the generated passage’s temporal as event A and the query's temporal as event B.
- Hard negatives:
    - Adjacent years or ranges (e.g., 2009 vs. query 2010).
    - Shifted intervals that look plausible but do not overlap (e.g., 2012–2014 for a query in 2010).
    - Misleading implicit cues that sound close but don’t satisfy the query (e.g., "shortly after 2015" vs. query "in 2010").
- Reuse: previously provided negatives should be reused where possible, but ensure variation and avoid duplicates.
- Quantity: at least 5 high-quality negatives per query, prioritizing quality over quantity.

5. Temporal extraction: The "temporal" field must contain the **full yet concise temporal expression as written** in the passage (e.g., "after July 2010", "from 2012 to 2014"), not just normalized dates, for "Explicit" and "Implicit" types. For instance, instead of "In 1906 he moved with his family to a farm", use "In 1906" and keep prepositions or context words that anchor the time, e.g., 'from', 'in', 'after', etc.'. For "TemporalAnswer" type, the "temporal" field must be an empty list.

6. Temporal type:
    - Must be "Explicit", "Implicit", or "TemporalAnswer". Be very careful with "Implicit" type, which is often confused. If the temporal contains a clear date/month/year, it is not implicit.

7 Query IDs:
   - All query split should have the original query id.

8. Output:
   - Only output valid JSON(s).
   - Do not explain, add comments, or include extra text, since your output will be parsed automatically.

### One-shot demonstration
Input:
{"query_id":0,"query":"On 2 July 2010 , after helping Setúbal avoid top-flight relegation , Barbosa was released by Porto , signing a three-year contract with S.C . Braga . Rarely used in the first months , he began gaining more playing time after the January 2011 departure of Matheus , who left for a team in Ukraine , and contributed four league goals in an eventual fourth-place finish .","positive_passages":[{"docid":2465, "text":"Hélder Barbosa played for which team from 2010 to 2013?"},{"docid":2465, "text":"Hélder Barbosa played for which team between June 2012 and October 2012?"}],"negative_passages":[{"docid":2465, "text":"Hélder Barbosa played for which team from 2002 to 2009?"},{"docid":2465, "text":"Hélder Barbosa played for which team from 2006 to 2009?"}]}
SUTIME tagger's output for query: [{'text': '2 July 2010', 'tid': 't3', 'type': 'DATE', 'value': '2010-07-02', 'span': [3, 14]}, {'text': 'three-year', 'tid': 't6', 'type': 'DURATION', 'value': 'P3Y', 'span': [111, 121]}, {'text': 'January 2011', 'tid': 't9', 'type': 'DATE', 'value': '2011-01', 'span': [229, 241]}]

Output:
{"query_id":0,"query":"On 2 July 2010, after helping Setúbal avoid top-flight relegation, Barbosa was released by Porto, signing a three-year contract with S.C. Braga.","temporal":["2 July 2010","three-year contract"],"positive_passages":[{"docid":2465,"text":"Which team did Hélder Barbosa play for from 2010 to 2013?","temporal":["from 2010 to 2013"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for on 2 July 2010?","temporal":["2 July 2010"],"allen_relation":"Equals","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa join after leaving Porto in July 2010?","temporal":["after leaving Porto in July 2010"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa sign a three-year contract with starting in 2010?","temporal":["starting in 2010"],"allen_relation":"Starts","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for between July 2010 and July 2013?","temporal":["between July 2010 and July 2013"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2465,"text":"When did Hélder Barbosa sign a contract with S.C. Braga.?","temporal":[],"allen_relation":"Empty","temporal_query_type":"TemporalAnswer"},{"docid":2465,"text":"How long did Hélder Barbosa's contract with S.C. Braga. last?","temporal":[],"allen_relation":"Empty","temporal_query_type":"TemporalAnswer"}],"negative_passages":[{"docid":2465,"text":"Which team did Hélder Barbosa play for from 2002 to 2009?","temporal":["from 2002 to 2009"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for between 2006 and 2009?","temporal":["between 2006 and 2009"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for after 2014?","temporal":["after 2014"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for during 2008?","temporal":["during 2008"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for prior to 2010?","temporal":["prior to 2010"],"allen_relation":"Before","temporal_query_type":"Explicit"}]}

{"query_id":0,"query":"Rarely used in the first months, he began gaining more playing time after the January 2011 departure of Matheus.","temporal":["after the January 2011 departure of Matheus"],"positive_passages":[{"docid":2465,"text":"Which team did Hélder Barbosa play for between January 2011 and December 2011?","temporal":["between January 2011 and December 2011"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for as of January 2011?","temporal":["as of January 2011"],"allen_relation":"Equals","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for after Matheus departed in January 2011?","temporal":["after Matheus departed in January 2011"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa contribute goals to during the 2011 season?","temporal":["during the 2011 season"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for throughout 2011?","temporal":["throughout 2011"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2465,"text":"When did Hélder Barbosa start getting more playing time?","temporal":[],"allen_relation":"Empty","temporal_query_type":"TemporalAnswer"},{"docid":2465,"text":"When did Hélder Barbosa begin gaining more playing time?","temporal":[],"allen_relation":"Empty","temporal_query_type":"TemporalAnswer"}],"negative_passages":[{"docid":2465,"text":"Which team did Hélder Barbosa play for before 2010?","temporal":["before 2010"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for after 2013?","temporal":["after 2013"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for in 2007?","temporal":["in 2007"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for prior to joining Braga?","temporal":["prior to joining Braga"],"allen_relation":"Before","temporal_query_type":"Implicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for during 2014?","temporal":["during 2014"],"allen_relation":"After","temporal_query_type":"Explicit"}]}
### End of one-shot demonstration

Note how one long "query" input is split into non-overlapping smaller queries with ideally one temporal expression each to avoid ambiguity in temporal information.

Now is your turn.
"""


# Start at 0.
#    - Increment by 1 for each new query or query split.


# You must also generate at least 5 positive and 5 negative QA-style passages to enable the model to learn temporal reasoning with Allen relations.
# If the input query is long and contains multiple temporal expressions, split it into smaller passages, where the shortest passage is a sentence that contains at least one temporal expression.

### This attempts to ask the LLM to automatically split and rewrite into multiple sentences with 1 temporal expression each, but it does not understand. It will always paste back the original query, even if I use the example query.
prompt_template = """You are a temporal annotation expert for information retrieval. We provide you with a query that may contain one or many temporal expressions and the corresponding SUTIME's output for your reference. Your job is to break down the query into multiple unique sub-queries with ideally one temporal expression each. For each sub-query, you must generate high-quality positive and negative passages for temporal contrastive learning.
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
- Empty: a special case for TemporalAnswer positive passages where there is no temporal expression.

* Temporal signals (examples of what can appear in queries or passages): before, prior to, until, after, following, since, in, on, as of, for duration, during, while, when, from...to..., between, by, up to, first, last, around, as soon as, as long as, for, over, all through, throughout, etc.

* Taxonomy of "positive_passages" and "negative_passages" with examples that you should generate:
- Explicit temporal constraints (i.e., always have clear temporal expressions that can be anchored to a specific datetime): "who won the state of texas in 2008?"; "what kind of government does iran have after 1979?".
- Implicit temporal constraints (i.e., events that cannot be anchored to specific datetime. The rule of thumb is: if there is date or month or year in the question, like "before the **2004** general election", it is not implicit temporal): "who was the president after jfk died?"; "what team did michael jordan play for after the bulls?".
- TemporalAnswer (i.e., Questions that inquire the datetime of an event instead of a general question with temporal constraints, usually starts with "when" or "what/which + day/date/month/year". There is no temporal expression in it.): "what year did the knicks win the championship?"; "when was the united nations founded?".


* Instructions:
1. You must output one or multiple valid JSONs, delimited by a newline, strictly following this pydantic schema:
{'$defs': {'AllenRelation': {'enum': ['Before', 'After', 'Meets', 'MetBy', 'Overlaps', 'OverlappedBy', 'Starts', 'StartedBy', 'During', 'Contains', 'Finishes', 'FinishedBy', 'Equals', 'Empty'], 'title': 'AllenRelation', 'type': 'string'}, 'Passage': {'properties': {'docid': {'title': 'Docid', 'type': 'integer'}, 'text': {'title': 'Text', 'type': 'string'}, 'temporal': {'items': {'type': 'string'}, 'title': 'Temporal', 'type': 'array'}, 'temporal_query_type': {'$ref': '#/$defs/TemporalQueryType'}, 'allen_relation': {'$ref': '#/$defs/AllenRelation'}}, 'required': ['docid', 'text', 'temporal', 'temporal_query_type', 'allen_relation'], 'title': 'Passage', 'type': 'object'}, 'TemporalQueryType': {'enum': ['Explicit', 'Implicit', 'TemporalAnswer'], 'title': 'TemporalQueryType', 'type': 'string'}}, 'properties': {'query_id': {'title': 'Query Id', 'type': 'integer'}, 'query': {'title': 'Query', 'type': 'string'}, 'temporal': {'items': {'type': 'string'}, 'title': 'Temporal', 'type': 'array'}, 'positive_passages': {'items': {'$ref': '#/$defs/Passage'}, 'title': 'Positive Passages', 'type': 'array'}, 'negative_passages': {'items': {'$ref': '#/$defs/Passage'}, 'title': 'Negative Passages', 'type': 'array'}}, 'required': ['query_id', 'query', 'temporal', 'positive_passages', 'negative_passages'], 'title': 'TemporalAnnotation', 'type': 'object'}.

2. "query_id": all rewritten sub-queries must be assigned the original query's "query_id".

3. “query":
- A query is a provided sentence that may contain one or many temporal expressions. If there is only one temporal expression, use the query as it is. If there are multiple temporal expressions (which is very common in timeline-style sentences), you must rewrite it into multiple unique and mutually exclusive sub-queries based on the original query's context, with each rewritten sub-query contains exactly one temporal expression from the original query. This is to avoid temporal ambiguity.
- For your reference, I will demonstrate with some example queries:
"query": "In 1919 Ehrensvärd was appointed Highest Commander of the Coastal Fleet and he was then commanding admiral and station commander in Karlskrona from 1919 to 1923 .". There are two temporal expressions in this query: "In 1919" and "from 1919 to 1923"; therefore, you must split this query into two sub-queries: Sub-query 1: "In 1919 Ehrensvärd was appointed Highest Commander of the Coastal Fleet.", the corresponding "temporal" is "In 1919"; Sub-query 2: "Ehrensvärd was then commanding admiral and station commander in Karlskrona from 1919 to 1923 .", the corresponding "temporal" is "from 1919 to 1923".
"query": "In August 2005, Brown joined Uruguayan giants Peñarol and in August 2006 he made his debut for fellow Uruguayans Tacuarembó." can be split into sub-queries:
Sub-query 1: "In August 2005, Brown joined Uruguayan giants Peñarol.", the corresponding "temporal" is "In August 2005"; Sub-query 2: "In August 2006 he made his debut for fellow Uruguayans Tacuarembó.", the corresponding "temporal" is "In August 2006".
"query": "In Summer 2010 he moved to fellow top flight side Austria Vienna, where he signed a contract until 2013." can be split into two queries: Sub-query 1: "In Summer 2010 he moved to fellow top flight side Austria Vienna", the corresponding "temporal" is "In Summer 2010"; Sub-query 2: "After moving to fellow top flight side Austria Vienna, he signed a contract until 2013.", the corresponding "temporal" is "until 2013".
- Each sub-query results in one fully-annotated JSON. You do not need to consider the original query anymore.
- The rule of thumb is to split the query into multiple sub-queries with a total number of at least equal to the number of SUTIME’s TIMEX3. However, please note that SUTIME only provides explicit temporal expressions and they are not perfect; therefore, you must double-check them and further detect additional explicit and implicit temporal expressions such as events.
- You must extract all events that are anchored to specific datetime (e.g., “2000 FA Cup Final”, “the 2007 election”, etc.).
- The "temporal" field should be extracted as written from the query's text and they must be concise, such as: "in April, 1906", "2 July 2010", "2004 general election", etc.

4. "positive_passages":
- Natural, QA-style questions with diverse phrasing that seek information from the query. Each question must contain exactly one extractable temporal expression, logically align with one of the query's temporal expressions.
- Based on the query's temporal expressions, you must generate "positive_passages" that cover all "TemporalQueryType", including 'Explicit', 'Implicit', and 'TemporalAnswer', when possible:
    - You MUST prioritize generate questions with explicit temporal constraints, like "in 2010", "from 2010 to 2015", etc. They must be logically align with the query’s temporal expression(s), such as be equals, overlap with, or be contained within the query’s temporal expression(s). If the query has multiple temporal expressions, the passage must logically align with at least one of them.
    - You should also prioritize generate questions with implicit temporal constraints, such as "after EVENT", "before EVENT". that must logically align with the query’s temporal expression(s) and diverse in Allen relations.
    - For both explicit and implicit "TemporalQueryType" questions, you must not use phrases like what/which date/day/month/year/time or when etc., that inquire time. You must not confuse this with TemporalAnswer questions.
    - Finally, you can generate TemporalAnswer (asking for datetime/duration/time-range of an EVENT, e.g., "When did EVENT happen?", "What time did he arrive?"). For this type of question, you must set: "TemporalQueryType": "TemporalAnswer", "allen_relation": "Empty", and "temporal": []
- "temporal" field: must extract all exact text spans of the temporal expressions that exist in the generated question (not normalized or paraphrased). Regarding explicit and implicit "TemporalQueryType", they must be concise temporal expressions as written, such as "after July 2010", "from 2012 to 2014", that are not just normalized dates. For instance, instead of "In 1906 he moved with his family to a farm", prefer the concise "In 1906" and keep prepositions or context words that anchor the time, e.g., 'from', 'in', 'after', etc. Regarding "TemporalAnswer" passages, the "temporal" field must be an empty list.
- Allen relation: must be correct, consider Passage = Event A and Query = Event B.
- “docid”: must remain the same as provided.
- Quantity: For each split/rewritten query, you must generate a list of 5 high-quality positive passages, prioritizing quality over quantity.

5. "negative_passages": Exactly the same as "positive_passages" in format, except that they generally have temporal constraints that are conflicted or do not exist in the query, or having irrelevant contexts. Based on the query's temporal expressions, you must generate "negative_passages" that cover all "TemporalQueryType", including "Explicit", "Implicit", and "TemporalAnswer", when possible. They must be hard and temporally-confused questions that belong to either of the following cases:
- Case 1: Questions with temporal constraints that DO NOT overlap with the query’s temporal expression(s): 
    - If the query is a span (e.g., 2005–2007), the temporal constraint must fall entirely outside (before 2005 or after 2007).
    - Open-ended intervals like “after 2010” or “before 2009” are valid only if they do not overlap with the query’s span.
    - Adjacent years or ranges (e.g., 2009 vs. query 2010).
    - Shifted intervals that look plausible but do not overlap (e.g., 2012–2014 for a query in 2010).
    - Misleading implicit cues that sound temporally close but are incorrect (e.g., “shortly after 2011” vs. query “in 2010”).
- Case 2: Questions with temporal constraints that overlap with the query’s temporal expression(s): Explicit/Implicit questions that uses the same temporal expression as the query; however, seeking information for irrelevant event or entity. For example, the query is about Ronaldo's career in 2010, but the question is asking for Messi's career in 2010.
- Quantity: For each split/rewritten query, you must generate a list of 5 high-quality negative passages, prioritizing quality over quantity.

6. "temporal_query_type": Must be "Explicit", "Implicit", or "TemporalAnswer". If the temporal contains a clear date/month/year, it is "Explicit", NOT "Implicit". If the pasage asks for which date/month/year of an event, it is "TemporalAnswer".
   
7. Final output: Only output valid JSON(s). Do not explain, add comments, or include extra text, since your output will be parsed automatically.

### One-shot demonstration
Input:
docid: 2465
query_id: 0
query: "On 2 July 2010 , after helping Setúbal avoid top-flight relegation , Barbosa was released by Porto , signing a three-year contract with S.C . Braga. Rarely used in the first months , he began gaining more playing time after the January 2011 departure of Matheus , who left for a team in Ukraine , and contributed four league goals in an eventual fourth-place finish ."
SUTIME's output: [{'text': '2 July 2010', 'tid': 't3', 'type': 'DATE', 'value': '2010-07-02', 'span': [3, 14]}, {'text': 'three-year', 'tid': 't6', 'type': 'DURATION', 'value': 'P3Y', 'span': [111, 121]}, {'text': 'January 2011', 'tid': 't9', 'type': 'DATE', 'value': '2011-01', 'span': [229, 241]}]

Output:
{"query_id":0,"query":"On 2 July 2010, after helping Setúbal avoid top-flight relegation, Barbosa was released by Porto, signing a three-year contract with S.C. Braga.","temporal":["2 July 2010","three-year contract"],"positive_passages":[{"docid":2465,"text":"Which team did Hélder Barbosa play for from 2010 to 2013?","temporal":["from 2010 to 2013"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for on 2 July 2010?","temporal":["2 July 2010"],"allen_relation":"Equals","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa join after leaving Porto in July 2010?","temporal":["after leaving Porto in July 2010"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa sign a three-year contract with?","temporal":["three-year contract"],"allen_relation":"Equals","temporal_query_type":"Explicit"},{"docid":2465,"text":"After being released by Porto, which team did Hélder Barbosa sign a contract with?","temporal":["After being released by Porto"],"allen_relation":"MetBy","temporal_query_type":"Implicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for between July 2010 and July 2013?","temporal":["between July 2010 and July 2013"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2465,"text":"When did Hélder Barbosa sign a contract with S.C. Braga.?","temporal":[],"allen_relation":"Empty","temporal_query_type":"TemporalAnswer"},{"docid":2465,"text":"How long did Hélder Barbosa's contract with S.C. Braga last?","temporal":[],"allen_relation":"Empty","temporal_query_type":"TemporalAnswer"}],"negative_passages":[{"docid":2465,"text":"Which team did Hélder Barbosa play for from 2002 to 2009?","temporal":["from 2002 to 2009"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for between 2006 and 2009?","temporal":["between 2006 and 2009"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for after 2014?","temporal":["after 2014"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for during 2008?","temporal":["during 2008"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for prior to 2010?","temporal":["prior to 2010"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Ronaldo play for in July 2010?","temporal":["in July 2010"],"allen_relation":"Equals","temporal_query_type":"Explicit"}]}

{"query_id":0,"query":"Rarely used in the first months, he began gaining more playing time after the January 2011 departure of Matheus , who left for a team in Ukraine , and contributed four league goals in an eventual fourth-place finish .","temporal":["after the January 2011 departure of Matheus"],"positive_passages":[{"docid":2465,"text":"Which team did Hélder Barbosa play for between January 2011 and December 2011?","temporal":["between January 2011 and December 2011"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for as of January 2011?","temporal":["as of January 2011"],"allen_relation":"Equals","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for after Matheus departed in January 2011?","temporal":["after Matheus departed in January 2011"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa contribute goals to during the 2011 season?","temporal":["during the 2011 season"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for throughout 2011?","temporal":["throughout 2011"],"allen_relation":"During","temporal_query_type":"Explicit"},{"docid":2465,"text":"When did Hélder Barbosa start getting more playing time?","temporal":[],"allen_relation":"Empty","temporal_query_type":"TemporalAnswer"},{"docid":2465,"text":"When did Hélder Barbosa begin gaining more playing time?","temporal":[],"allen_relation":"Empty","temporal_query_type":"TemporalAnswer"}],"negative_passages":[{"docid":2465,"text":"Which team did Hélder Barbosa play for before 2010?","temporal":["before 2010"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for after 2013?","temporal":["after 2013"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for in 2007?","temporal":["in 2007"],"allen_relation":"Before","temporal_query_type":"Explicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for prior to joining Braga?","temporal":["prior to joining Braga"],"allen_relation":"Before","temporal_query_type":"Implicit"},{"docid":2465,"text":"Which team did Hélder Barbosa play for during 2014?","temporal":["during 2014"],"allen_relation":"After","temporal_query_type":"Explicit"},{"docid":2465,"text":"After the January 2011 departure of Matheus, did Ronaldo get more playing time?","temporal":["After the January 2011 departure of Matheus"],"allen_relation":"Equals","temporal_query_type":"Explicit"}]}
### End of one-shot demonstration"""


final_prompt_template = """You are a temporal annotation expert for information retrieval. We provide you with a query that may contain one or many temporal expressions and the corresponding SUTIME's output for your reference. Your job is to generate high-quality positive passages and negative passages for temporal contrastive learning.
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