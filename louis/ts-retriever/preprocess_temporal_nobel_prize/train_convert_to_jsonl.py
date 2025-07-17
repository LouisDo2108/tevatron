from pdb import set_trace as st

import msgspec
from tqdm.auto import tqdm

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

def read_json(file_path, jsonl=False):
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


def convert_corpus_json_to_jsonl(output_file: str):

    train = read_json(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time-sensitive-qa/annotated/annotated_train.json",
        jsonl=False,
    )
    dev = read_json(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time-sensitive-qa/annotated/annotated_dev.json",
        jsonl=False,
    )
    test = read_json(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time-sensitive-qa/annotated/annotated_test.json",
        jsonl=False,
    )

    train_corpus = {" ".join(v["paras"]) for v in train}
    dev_corpus = {" ".join(v["paras"]) for v in dev}
    test_corpus = {" ".join(v["paras"]) for v in test}
    final_corpus = train_corpus.union(dev_corpus).union(test_corpus)
    
    output_jsonl = []
    for ix, v in enumerate(final_corpus):
        output_jsonl.append(
            {
                "docid": ix,
                "text": v,
            }
        )
    write_json(output_file, output_jsonl, jsonl=True)


def convert_to_tevatron_msmarco_jsonl(input_file: str, output_file: str):

    docid_to_doc_dict = read_json(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/train/corpus.jsonl",
        jsonl=True
    )
    docid_to_doc_dict = {v["docid"]: v["text"] for v in docid_to_doc_dict}

    from collections import defaultdict

    inverted_index = defaultdict(set)

    for docid, doc in docid_to_doc_dict.items():
        tokens = doc.lower().split()  # or use nltk.word_tokenize for better tokenization
        for token in set(tokens):  # Use set to avoid duplicates
            inverted_index[token].add(docid)

    input_jsonl = read_json(input_file, jsonl=True)
    output_jsonl = []

    for ix, v in tqdm(enumerate(input_jsonl)):
        query = v["question"]
        positive_ctxs = v["positive_ctxs"]
        negative_ctxs = v["negative_ctxs"]
        docid = None

        query_terms = query.lower().split()
        # Get sets of doc IDs for each term
        docid_sets = [inverted_index.get(term, set()) for term in query_terms]

        # Intersect sets to find common docs
        if docid_sets:
            candidate_docids = set.intersection(*docid_sets)
        else:
            candidate_docids = set()

        # Optionally, check if the full query substring is in the doc
        matching_docs = [
            docid for docid in candidate_docids if query in docid_to_doc_dict[docid]
        ]

        if len(matching_docs) == 1:
            docid = matching_docs[0]
        else:
            docid = matching_docs
            print(f"Query '{query}' matched {len(matching_docs)} documents: {matching_docs}")

        positive_passages = []
        for doc in positive_ctxs:
            if isinstance(docid, list):
                for _docid in matching_docs:
                    positive_passages.append(
                        {
                            "docid": _docid,
                            "text": doc["text"],
                        }
                    )
                print("Multiple matching documents found for positive passages.")
            else:
                positive_passages.append(
                    {
                        "docid": docid,
                        "text": doc["text"],
                    }
                )

        negative_passages = []
        for doc in negative_ctxs:
            if isinstance(docid, list):
                for _docid in matching_docs:
                    negative_passages.append(
                        {
                            "docid": _docid,
                            "text": doc["text"],
                        }
                    )
                print(
                    "Multiple matching documents found for negative passages."
                )
            else:
                negative_passages.append(
                    {
                        "docid": docid,
                        "text": doc["text"],
                    }
                )

        output_jsonl.append({
                "query_id": ix,
                "query": query,
                "positive_passages": positive_passages,
                "negative_passages": negative_passages,
        })
        
    write_json(output_file, output_jsonl, jsonl=True)

    with open(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/train/chunked/corpus.jsonl",
        "rb",
    ) as infile:
        docid_to_doc_dict = decoder.decode_lines(infile.read())
    docid_to_doc_dict = {v["docid"]: v["text"] for v in docid_to_doc_dict}

    from collections import defaultdict

    inverted_index = defaultdict(set)

    for docid, doc in docid_to_doc_dict.items():
        tokens = (
            doc.lower().split()
        )  # or use nltk.word_tokenize for better tokenization
        for token in set(tokens):  # Use set to avoid duplicates
            inverted_index[token].add(docid)

    output_jsonl = []
    with open(input_file, "rb") as infile:
        input_jsonl = decoder.decode_lines(infile.read())

    with open(output_file, "wb") as outfile:
        for ix, v in tqdm(enumerate(input_jsonl)):
            query = v["question"]
            positive_ctxs = v["positive_ctxs"]
            negative_ctxs = v["negative_ctxs"]
            docid = None

            query_terms = query.lower().split()
            # Get sets of doc IDs for each term
            docid_sets = [inverted_index.get(term, set()) for term in query_terms]

            # Intersect sets to find common docs
            if docid_sets:
                candidate_docids = set.intersection(*docid_sets)
            else:
                candidate_docids = set()

            # Optionally, check if the full query substring is in the doc
            matching_docs = [
                docid for docid in candidate_docids if query in docid_to_doc_dict[docid]
            ]

            if len(matching_docs) == 1:
                docid = matching_docs[0]

            if len(matching_docs) == 0:
                print(
                    f"Query '{query}' matched {len(matching_docs)} documents: {matching_docs}"
                )
                # st()
                # for cand_docid in candidate_docids:
                #     cand_doc = docid_to_doc_dict[cand_docid]
                #     cand_doc_term_set = set(cand_doc.lower().split())


                #     union = set(query_terms).union(cand_doc_term_set)

                #     if len(query_terms) >= len(union) * 0.9:
                #         print("Partially matched")

            if len(matching_docs) > 1:
                print(
                    f"Query '{query}' matched {len(matching_docs)} documents: {matching_docs}"
                )

            positive_passages = []
            for doc in positive_ctxs:
                if isinstance(docid, list):
                    for _docid in matching_docs:
                        positive_passages.append(
                            {
                                "docid": _docid,
                                "text": doc["text"],
                            }
                        )
                    print("Multiple matching documents found for positive passages.")
                else:
                    positive_passages.append(
                        {
                            "docid": docid,
                            "text": doc["text"],
                        }
                    )

            negative_passages = []
            for doc in negative_ctxs:
                if isinstance(docid, list):
                    for _docid in matching_docs:
                        negative_passages.append(
                            {
                                "docid": _docid,
                                "text": doc["text"],
                            }
                        )
                    print("Multiple matching documents found for negative passages.")
                else:
                    negative_passages.append(
                        {
                            "docid": docid,
                            "text": doc["text"],
                        }
                    )

            output_jsonl.append(
                {
                    "query_id": ix,
                    "query": query,
                    "positive_passages": positive_passages,
                    "negative_passages": negative_passages,
                }
            )
        outfile.write(encoder.encode_lines(output_jsonl))


if __name__ == "__main__":

    # convert_corpus_json_to_jsonl(
    #     output_file="/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/train/corpus.jsonl"
    # )

    input_file = "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/train/original/contriever_finetune_eval_v3.jsonl"
    output_file = "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/train/chunked/dev.jsonl"

    convert_to_tevatron_msmarco_jsonl(input_file, output_file)
