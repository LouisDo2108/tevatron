import logging
from dataclasses import dataclass
from pdb import set_trace as st
from dataclasses import dataclass, field
import torch.nn.functional as F

from tevatron.retriever.collator import EncodeCollator, TrainCollator

logger = logging.getLogger(__name__)

temporal_max_length = 16


@dataclass
class UnsupervisedMAdaptorCollator(EncodeCollator):

    def __call__(self, features):
        """
        Collate function for encoding.
        :param features: list of (id, text, image) tuples
        but in this case, it's just image is None
        """
        query = [x[0] for x in features]  # Dummy lists
        d = [x[1] for x in features]

        content_docids = [x[0] for x in d]
        content_chunkids = [x[1] for x in d]
        docs = [x[2] for x in d]
        max_length = (
            self.data_args.query_max_len
            if self.data_args.encode_is_query
            else self.data_args.passage_max_len
        )
        collated_inputs = self.tokenizer(
            docs,
            padding=True,
            truncation=True,
            max_length=(
                max_length - 1 if self.data_args.append_eos_token else max_length
            ),
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        return query, (content_docids, content_chunkids, collated_inputs)


@dataclass
class UnsupervisedTemporalMAdaptorCollator(UnsupervisedMAdaptorCollator):

    def __call__(self, features):
        """
        Collate function for encoding.
        :param features: list of (id, text, image) tuples
        but in this case, it's just image is None
        """
        query = [x[0] for x in features]  # Dummy lists
        d = [x[1] for x in features]

        content_docids = [x[0] for x in d]
        content_chunkids = [x[1] for x in d]
        docs = [x[2] for x in d]
        temporal = [x[3] for x in d]

        max_length = (
            self.data_args.query_max_len
            if self.data_args.encode_is_query
            else self.data_args.passage_max_len
        )
        collated_docs = self.tokenizer(
            docs,
            padding=True,
            truncation=True,
            max_length=(
                max_length - 1 if self.data_args.append_eos_token else max_length
            ),
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        collated_temporal = self.tokenizer(
            temporal,
            padding=True,
            truncation=True,
            max_length=(
                max_length - 1 if self.data_args.append_eos_token else max_length
            ),
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        return query, (content_docids, content_chunkids, collated_docs, collated_temporal)


@dataclass
class SupervisedMAdaptorCollator(TrainCollator):

    def __call__(self, features):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        all_queries = [f[0] for f in features]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])

        all_corpus_docid = []
        all_corpus_chunkid = []
        all_corpus = []

        for element in (f[2] for f in features):
            for chunkid, docid, doc in element:
                all_corpus_docid.append(chunkid)
                all_corpus_chunkid.append(docid)
                all_corpus.append(doc)

        all_queries = [q[0] for q in all_queries]
        all_passages = [p[0] for p in all_passages]

        q_collated = self.tokenizer(
            all_queries,
            padding=True,
            truncation=True,
            max_length=(
                self.data_args.query_max_len - 1
                if self.data_args.append_eos_token
                else self.data_args.query_max_len
            ),
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            all_passages,
            padding=True,
            truncation=True,
            max_length=(
                self.data_args.passage_max_len - 1
                if self.data_args.append_eos_token
                else self.data_args.passage_max_len
            ),
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
        )
        corpus_collated = self.tokenizer(
            all_corpus,
            padding=True,
            truncation=True,
            max_length=(
                self.data_args.passage_max_len - 1
                if self.data_args.append_eos_token
                else self.data_args.passage_max_len
            ),
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
        )

        # q_collated = self.tokenizer.pad(
        #     q_collated,
        #     padding=True,
        #     pad_to_multiple_of=self.data_args.pad_to_multiple_of,
        #     return_attention_mask=True,
        #     return_tensors="pt",
        # )
        # d_collated = self.tokenizer.pad(
        #     d_collated,
        #     padding=True,
        #     pad_to_multiple_of=self.data_args.pad_to_multiple_of,
        #     return_attention_mask=True,
        #     return_tensors="pt",
        # )
        # corpus_collated = self.tokenizer.pad(
        #     corpus_collated,
        #     padding=True,
        #     pad_to_multiple_of=self.data_args.pad_to_multiple_of,
        #     return_attention_mask=True,
        #     return_tensors="pt",
        # )
        return q_collated, (d_collated, all_corpus_docid, all_corpus_chunkid, corpus_collated)


@dataclass
class NaiveTemporalCollator(TrainCollator):
    """
    A naive collator that consider the temporal expressions from the text as sentences.
    """

    def __call__(self, features):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        all_queries = [f[0] for f in features]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])

        # all_corpus_docid = []
        # all_corpus_chunkid = []
        # all_corpus = []

        # for element in (f[2] for f in features):
        #     for chunkid, docid, doc in element:
        #         all_corpus_docid.append(chunkid)
        #         all_corpus_chunkid.append(docid)
        #         all_corpus.append(doc)

        all_queries = [q[0] for q in all_queries]

        all_passages = []
        all_passages_temporal = []
        for training_sample in [f[1] for f in features]:
            for passage, passage_temporal in training_sample:
                all_passages.append(passage)
                all_passages_temporal.append(" ".join(passage_temporal)) # comma separated temporal expressions -> concatenate into one 

        q_collated = self.tokenizer(
            all_queries,
            padding=True,
            truncation=True,
            max_length=(
                self.data_args.query_max_len - 1
                if self.data_args.append_eos_token
                else self.data_args.query_max_len
            ),
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            all_passages,
            padding=True,
            truncation=True,
            max_length=(
                self.data_args.passage_max_len - 1
                if self.data_args.append_eos_token
                else self.data_args.passage_max_len
            ),
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
        )
        d_temporal_collated = self.tokenizer(
            all_passages_temporal,
            padding=True,
            truncation=True,
            max_length=128,
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
        )
        return q_collated, (
            d_collated,
            d_temporal_collated,
        )


@dataclass
class NaiveTemporalv2Collator(TrainCollator):
    """
    An improved collator based on NaiveTemporalCollator. It nows extract the temporal expressions span in the queries.
    """

    def __call__(self, features):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        all_queries = [f[0] for f in features]
        all_queries = [q[0] for q in all_queries]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])

        all_passages = []
        char_spans = []  # flat list of (start_char, end_char) per passage
        span_counts = []  # how many spans per passage, to reconstruct later

        # Step 1: Collect passages and raw char spans
        for training_sample in [f[1] for f in features]:
            for passage, passage_temporal in training_sample:
                all_passages.append(passage)
                curr_spans = []
                for t in passage_temporal:
                    start = passage.find(t)
                    if start != -1:
                        curr_spans.append((start, start + len(t)))
                char_spans.extend(curr_spans)
                span_counts.append(len(curr_spans))

        q_collated = self.tokenizer(
            all_queries,
            padding=True,
            truncation=True,
            max_length=(
                self.data_args.query_max_len - 1
                if self.data_args.append_eos_token
                else self.data_args.query_max_len
            ),
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            all_passages,
            padding=True,
            truncation=True,
            max_length=(
                self.data_args.passage_max_len - 1
                if self.data_args.append_eos_token
                else self.data_args.passage_max_len
            ),
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
            return_offsets_mapping=True,  # Step 2: Tokenize with offset mapping
        )

        # Step 3: Map char spans to token spans
        temporal_token_spans = []
        char_span_cursor = 0

        for ix, num_spans in enumerate(span_counts):
            offsets = d_collated["offset_mapping"][ix]
            token_spans = []

            for _ in range(num_spans):
                char_start, char_end = char_spans[char_span_cursor]
                token_indices = [
                    i for i, (s, e) in enumerate(offsets)
                    if s != 0 or e != 0  # skip special tokens
                    if e > char_start and s < char_end
                ]

                if token_indices:
                    token_spans.append((token_indices[0], token_indices[-1]))

                char_span_cursor += 1

            temporal_token_spans.append(token_spans)

        d_collated.pop("offset_mapping", None)  # Remove offset mapping
        return q_collated, (
            d_collated,
            temporal_token_spans,
        )


@dataclass
class NaiveTemporalv3Collator(TrainCollator):
    """
    An improved collator based on NaiveTemporalv2Collator. It also returns the input_ids of the temporal expressions for temporal reconstruction.
    """

    def __call__(self, features):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        all_queries = [f[0] for f in features]
        all_queries = [q[0] for q in all_queries]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])

        if len(features[0]) == 2: # No extracted temporal
            all_passages = [p[0] for p in all_passages]
        elif len(features[0]) == 3: # With extracted temporal
            all_passages = []
            char_spans = []  # flat list of (start_char, end_char) per passage
            span_counts = []  # how many spans per passage, to reconstruct later

            # Step 1: Collect passages and raw char spans
            for training_sample in [f[1] for f in features]:
                for passage, passage_temporal in training_sample:
                    all_passages.append(passage)
                    curr_spans = []
                    for t in passage_temporal:
                        start = passage.lower().find(t.lower()) # There are cases where the extracted temporal expressions are in upper case and the passage is in lower case
                        if start != -1:
                            curr_spans.append((start, start + len(t)))
                    
                    # For debugging      
                    # if len(curr_spans) == 0:
                    #     print(passage, passage_temporal)
                    #     st()
                        # print(passage[start:start+len(t)])
                    char_spans.extend(curr_spans)
                    span_counts.append(len(curr_spans))

        q_collated = self.tokenizer(
            all_queries,
            padding=True,
            truncation=True,
            max_length=(
                self.data_args.query_max_len - 1
                if self.data_args.append_eos_token
                else self.data_args.query_max_len
            ),
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            all_passages,
            padding=True,
            truncation=True,
            max_length=(
                self.data_args.passage_max_len - 1
                if self.data_args.append_eos_token
                else self.data_args.passage_max_len
            ),
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
            return_offsets_mapping=True,  # Step 2: Tokenize with offset mapping
        )

        if len(features[0]) == 2:
            # Fall back to encoder dataset collater used for evaluation
            d_collated.pop("offset_mapping", None)  # Remove offset mapping
            return q_collated, d_collated

        # Step 3: Map char spans to token spans
        temporal_token_spans = []
        temporal_tokens_input_ids = []
        char_span_cursor = 0

        for ix, num_spans in enumerate(span_counts):
            offsets = d_collated["offset_mapping"][ix]
            input_ids = d_collated["input_ids"][ix]
            token_spans = []
            token_input_ids = []

            for _ in range(num_spans):
                char_start, char_end = char_spans[char_span_cursor]
                token_indices = [
                    i for i, (s, e) in enumerate(offsets)
                    if s != 0 or e != 0  # skip special tokens
                    if e > char_start and s < char_end
                ]

                if token_indices:
                    token_spans.append((token_indices[0], token_indices[-1]))
                    _token_input_ids = input_ids[token_indices]
                    token_input_ids.append(
                        F.pad(_token_input_ids, (0, temporal_max_length-len(_token_input_ids)), value=0)
                    )
                # else:
                #     token_input_ids.append(torch.ones(16)*-100)
                char_span_cursor += 1

            temporal_token_spans.append(token_spans) 
            temporal_tokens_input_ids.append(token_input_ids)

        d_collated.pop("offset_mapping", None)  # Remove offset mapping
        return q_collated, (
            d_collated,
            temporal_token_spans, # Spans for extracting the corresponding temporal tokens
            temporal_tokens_input_ids, # Labels for reconstruction loss
        )

# @dataclass
# class SentenceTransformerStyleTemporalCollator(NaiveTemporalv2Collator):

#     valid_label_columns: list[str] = field(default_factory=lambda: ["label", "score"])

#     def __call__(self, features):
#         """
#         Collate function for training.
#         :param features: list of (query, passages) tuples
#         :return: tokenized query_ids, passage_ids
#         """
#         all_queries = [f[0] for f in features]
#         all_passages = []
#         for f in features:
#             all_passages.extend(f[1])

#         all_queries = [q[0] for q in all_queries]

#         all_passages = []
#         char_spans = []  # flat list of (start_char, end_char) per passage
#         span_counts = []  # how many spans per passage, to reconstruct later

#         # Step 1: Collect passages and raw char spans (faster than nested temp lists)
#         for training_sample in [f[1] for f in features]:
#             for passage, passage_temporal in training_sample:
#                 all_passages.append(passage)
#                 curr_spans = []
#                 for t in passage_temporal:
#                     start = passage.find(t)
#                     if start != -1:
#                         curr_spans.append((start, start + len(t)))
#                 char_spans.extend(curr_spans)
#                 span_counts.append(len(curr_spans))

#         q_collated = self.tokenizer(
#             all_queries,
#             padding=True,
#             truncation=True,
#             max_length=(
#                 self.data_args.query_max_len - 1
#                 if self.data_args.append_eos_token
#                 else self.data_args.query_max_len
#             ),
#             return_attention_mask=True,
#             return_token_type_ids=False,
#             add_special_tokens=True,
#             pad_to_multiple_of=self.data_args.pad_to_multiple_of,
#             return_tensors="pt",
#         )
#         d_collated = self.tokenizer(
#             all_passages,
#             padding=True,
#             truncation=True,
#             max_length=(
#                 self.data_args.passage_max_len - 1
#                 if self.data_args.append_eos_token
#                 else self.data_args.passage_max_len
#             ),
#             return_attention_mask=True,
#             return_token_type_ids=False,
#             add_special_tokens=True,
#             pad_to_multiple_of=self.data_args.pad_to_multiple_of,
#             return_tensors="pt",
#             return_offsets_mapping=True,  # Step 2: Tokenize with offset mapping
#         )

#         # Step 3: Map char spans to token spans
#         temporal_token_spans = []
#         char_span_cursor = 0

#         for ix, num_spans in enumerate(span_counts):
#             offsets = d_collated["offset_mapping"][ix]
#             token_spans = []

#             for _ in range(num_spans):
#                 char_start, char_end = char_spans[char_span_cursor]
#                 token_indices = [
#                     i
#                     for i, (s, e) in enumerate(offsets)
#                     if s != 0 or e != 0  # skip special tokens
#                     if e > char_start and s < char_end
#                 ]

#                 if token_indices:
#                     token_spans.append((token_indices[0], token_indices[-1]))

#                 char_span_cursor += 1

#             temporal_token_spans.append(token_spans)

#         d_collated.pop("offset_mapping", None)  # Remove offset mapping
#         return q_collated, (
#             d_collated,
#             temporal_token_spans,
#         )
