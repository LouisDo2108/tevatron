import logging
from dataclasses import dataclass
from pdb import set_trace as st
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

from tevatron.retriever.collator import EncodeCollator, TrainCollator

logger = logging.getLogger(__name__)

temporal_query_type_class_id = {
    "Explicit": 0,
    "Implicit": 1,
    "TemporalAnswer": 2,
}

allen_relation_class_id = {
    "Before": 0,
    "After": 1,
    "Meets": 2,
    "MetBy": 3,
    "Overlaps": 4,
    "OverlappedBy": 5,
    "Starts": 6,
    "StartedBy": 7,
    "During": 8,
    "Contains": 9,
    "Finishes": 10,
    "FinishedBy": 11,
    "Equals": 12,
    "Empty": 13,
}


@dataclass
class TemporalCollator(TrainCollator):
    """
    An improved collator based on NaiveTemporalCollator. It nows extract the temporal expressions span in the queries.
    """

    def __call__(self, features):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        if len(features[0]) != 3:
            all_queries = [f[0][0] for f in features]
            all_passages = [p[0] for f in features for p in f[1]]
            q_collated = self.tokenizer(
                all_queries,
                padding=True,
                truncation=True,
                max_length=(
                    self.data_args.query_max_len - 1
                    if self.data_args.append_eos_token
                    else self.data_args.query_max_len
                ),
                pad_to_multiple_of=self.data_args.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
                return_token_type_ids=False,
                add_special_tokens=True,
                padding_side=self.data_args.padding_side,
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
                pad_to_multiple_of=self.data_args.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
                return_token_type_ids=False,
                add_special_tokens=True,
                padding_side=self.data_args.padding_side,
            )

            return q_collated, d_collated

        all_query_list_str = [f[0] for f in features]
        all_query_temporal_list_list_str = [f[1] for f in features]

        all_passages_str = []
        all_passages_temporal_list_list_str = []
        all_passages_temporal_query_type_list_str = []
        all_passages_allen_relation_list_str = []

        for f in features:
            for p in f[2]:
                all_passages_str.append(p[0])
                all_passages_temporal_list_list_str.append(p[1])
                all_passages_temporal_query_type_list_str.append(p[2])
                all_passages_allen_relation_list_str.append(p[3])

        query_char_spans_list, query_span_counts_list = self.get_temporal_char_spans(
            all_query_list_str, all_query_temporal_list_list_str
        )

        passage_char_spans_list, passage_span_counts_list = (
            self.get_temporal_char_spans(
                all_passages_str, all_passages_temporal_list_list_str
            )
        )

        # Process the query side
        q_collated_list = self.tokenizer(
            all_query_list_str,
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
            return_offsets_mapping=True,
            padding_side=self.data_args.padding_side,
        )

        # Process the passage side
        d_collated_list = self.tokenizer(
            all_passages_str,
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
            padding_side=self.data_args.padding_side,
        )

        # Step 3: Map char spans to token spans
        # Including Spans for extracting the corresponding temporal tokens and Labels for reconstruction loss
        query_temporal_token_spans_list = (
            self.get_temporal_token_spans(
                query_char_spans_list,
                query_span_counts_list,
                q_collated_list,
            )
        )

        passage_temporal_token_spans_list = (
            self.get_temporal_token_spans(
                passage_char_spans_list,
                passage_span_counts_list,
                d_collated_list,
            )
        )

        q_collated_list.pop("offset_mapping", None)  # Remove offset mapping
        d_collated_list.pop("offset_mapping", None)  # Remove offset mapping

        return (
            q_collated_list,
            query_temporal_token_spans_list,
            None,
        ), (
            d_collated_list,
            passage_temporal_token_spans_list,
            None,
            torch.as_tensor(
                [
                    temporal_query_type_class_id[x]
                    for x in all_passages_temporal_query_type_list_str
                ]
            ),
            torch.as_tensor(
                [
                    allen_relation_class_id[x]
                    for x in all_passages_allen_relation_list_str
                ]
            ),
        )

    def get_temporal_token_spans(
        self,
        passage_char_spans_list,
        passage_span_counts_list,
        d_collated,
    ):
        temporal_token_spans = []
        char_span_cursor = 0

        for ix, num_spans in enumerate(passage_span_counts_list):
            offsets = d_collated["offset_mapping"][ix]
            token_spans = []

            for _ in range(num_spans):
                char_start, char_end = passage_char_spans_list[char_span_cursor]
                token_indices = [
                    i
                    for i, (s, e) in enumerate(offsets)
                    if s != 0 or e != 0  # skip special tokens
                    if e > char_start and s < char_end
                ]

                if token_indices:
                    token_spans.append((token_indices[0], token_indices[-1]))
                char_span_cursor += 1

            temporal_token_spans.append(token_spans)

        return temporal_token_spans

    def get_temporal_char_spans(
        self, all_passages_str, all_passages_temporal_list_list_str
    ):
        passage_char_spans_list = []  # flat list of (start_char, end_char) per passage
        passage_span_counts_list = (
            []
        )  # how many spans per passage, to reconstruct later
        # Step 1: Collect passages and raw char spans
        for passage, passage_temporal in zip(
            all_passages_str, all_passages_temporal_list_list_str
        ):
            curr_spans = []
            passage = passage.lower()
            for t in passage_temporal:
                start = passage.find(
                    t.lower()
                )  # There are cases where the extracted temporal expressions are in upper case and the passage is in lower case
                if start != -1:
                    curr_spans.append((start, start + len(t)))

            # For debugging
            # if len(curr_spans) == 0:
            #     print(passage, passage_temporal)
            #     st()
            # print(passage[start:start+len(t)])
            # # This is only for reconstruction loss no need to be exactly the same as the batch size
            # if curr_spans:
            passage_char_spans_list.extend(curr_spans)
            passage_span_counts_list.append(len(curr_spans))
        return passage_char_spans_list, passage_span_counts_list


@dataclass
class TemporalReconPassageSideCollator(TrainCollator):
    """
    An improved collator based on NaiveTemporalv2Collator. 
    It also returns the input_ids of the temporal expressions for temporal reconstruction.
    """

    def __call__(self, features):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """

        temporal_max_length = 16 # Might need to increases this in the case of very long temporal

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
            padding_side=self.data_args.padding_side
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
            padding_side=self.data_args.padding_side
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


@dataclass
class TemporalReconCollator(TrainCollator):
    """
    An improved collator based on NaiveTemporalv2Collator. It also returns the input_ids of the temporal expressions for temporal reconstruction.
    """

    def __call__(self, features):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        if len(features[0]) != 3:
            all_queries = [f[0][0] for f in features]
            all_passages = [p[0] for f in features for p in f[1]]
            q_collated = self.tokenizer(
                all_queries,
                padding=True,
                truncation=True,
                max_length=(
                    self.data_args.query_max_len - 1
                    if self.data_args.append_eos_token
                    else self.data_args.query_max_len
                ),
                pad_to_multiple_of=self.data_args.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
                return_token_type_ids=False,
                add_special_tokens=True,
                padding_side=self.data_args.padding_side,
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
                pad_to_multiple_of=self.data_args.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
                return_token_type_ids=False,
                add_special_tokens=True,
                padding_side=self.data_args.padding_side,
            )

            return q_collated, d_collated

        all_query_list_str = [f[0] for f in features]
        all_query_temporal_list_list_str = [f[1] for f in features]

        all_passages_str = []
        all_passages_temporal_list_list_str = []
        all_passages_temporal_query_type_list_str = []
        all_passages_allen_relation_list_str = []

        for f in features:
            for p in f[2]:
                all_passages_str.append(p[0])
                all_passages_temporal_list_list_str.append(p[1])
                all_passages_temporal_query_type_list_str.append(p[2])
                all_passages_allen_relation_list_str.append(p[3])

        query_char_spans_list, query_span_counts_list = self.get_temporal_char_spans(
            all_query_list_str, all_query_temporal_list_list_str
        )

        passage_char_spans_list, passage_span_counts_list = (
            self.get_temporal_char_spans(
                all_passages_str, all_passages_temporal_list_list_str
            )
        )

        # Process the query side
        q_collated_list = self.tokenizer(
            all_query_list_str,
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
            return_offsets_mapping=True,
            padding_side=self.data_args.padding_side,
        )

        # Process the passage side
        d_collated_list = self.tokenizer(
            all_passages_str,
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
            padding_side=self.data_args.padding_side,
        )

        # Step 3: Map char spans to token spans
        # Including Spans for extracting the corresponding temporal tokens and Labels for reconstruction loss
        query_temporal_token_spans_list, query_temporal_tokens_input_ids_list = (
            self.get_temporal_token_spans(
                query_char_spans_list,
                query_span_counts_list,
                q_collated_list,
                is_passage=False,
            )
        )

        passage_temporal_token_spans_list, passage_temporal_tokens_input_ids_list = (
            self.get_temporal_token_spans(
                passage_char_spans_list,
                passage_span_counts_list,
                d_collated_list,
                is_passage=True,
            )
        )

        q_collated_list.pop("offset_mapping", None)  # Remove offset mapping
        d_collated_list.pop("offset_mapping", None)  # Remove offset mapping

        return (
            q_collated_list,
            query_temporal_token_spans_list,
            query_temporal_tokens_input_ids_list,
        ), (
            d_collated_list,
            passage_temporal_token_spans_list,
            passage_temporal_tokens_input_ids_list,
            torch.as_tensor(
                [
                    temporal_query_type_class_id[x]
                    for x in all_passages_temporal_query_type_list_str
                ]
            ),
            torch.as_tensor(
                [
                    allen_relation_class_id[x]
                    for x in all_passages_allen_relation_list_str
                ]
            ),
        )

    def get_temporal_token_spans(
        self,
        passage_char_spans_list,
        passage_span_counts_list,
        d_collated,
        num_neg=2,
        temporal_max_length=16,
        is_passage=False,
    ):
        temporal_token_spans = []
        temporal_tokens_input_ids = []
        char_span_cursor = 0

        token_input_ids = []
        for ix, num_spans in enumerate(passage_span_counts_list):
            offsets = d_collated["offset_mapping"][ix]
            input_ids = d_collated["input_ids"][ix]
            token_spans = []

            for _ in range(num_spans):
                char_start, char_end = passage_char_spans_list[char_span_cursor]
                token_indices = [
                    i
                    for i, (s, e) in enumerate(offsets)
                    if s != 0 or e != 0  # skip special tokens
                    if e > char_start and s < char_end
                ]

                if token_indices:
                    token_spans.append((token_indices[0], token_indices[-1]))
                    _token_input_ids = input_ids[token_indices]
                    token_input_ids.append(
                        F.pad(
                            _token_input_ids,
                            (0, temporal_max_length - len(_token_input_ids)),
                            value=0,
                        )
                    )
                # else:
                #     token_input_ids.append(torch.ones(16)*-100)
                char_span_cursor += 1

            temporal_token_spans.append(token_spans)

            if not is_passage:
                temporal_tokens_input_ids.append(token_input_ids)
                token_input_ids = []
            elif is_passage and (ix + 1) % num_neg == 0:
                temporal_tokens_input_ids.append(token_input_ids)
                token_input_ids = []

        return temporal_token_spans, temporal_tokens_input_ids

    def get_temporal_char_spans(
        self, all_passages_str, all_passages_temporal_list_list_str
    ):
        passage_char_spans_list = []  # flat list of (start_char, end_char) per passage
        passage_span_counts_list = (
            []
        )  # how many spans per passage, to reconstruct later
        # Step 1: Collect passages and raw char spans
        for passage, passage_temporal in zip(
            all_passages_str, all_passages_temporal_list_list_str
        ):
            curr_spans = []
            passage = passage.lower()
            for t in passage_temporal:
                start = passage.find(
                    t.lower()
                )  # There are cases where the extracted temporal expressions are in upper case and the passage is in lower case
                if start != -1:
                    curr_spans.append((start, start + len(t)))

            # For debugging
            # if len(curr_spans) == 0:
            #     print(passage, passage_temporal)
            #     st()
            # print(passage[start:start+len(t)])
            # # This is only for reconstruction loss no need to be exactly the same as the batch size
            # if curr_spans:
            passage_char_spans_list.extend(curr_spans)
            passage_span_counts_list.append(len(curr_spans))
        return passage_char_spans_list, passage_span_counts_list


@dataclass
class TempRetrieverCollator(TrainCollator):
    """
    An improved collator based on NaiveTemporalv2Collator. It also returns the input_ids of the temporal expressions for temporal reconstruction.
    """

    def __call__(self, features):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        if len(features[0]) != 3:
            all_queries = [f[0] for f in features]
            all_passages = []
            for f in features:
                all_passages.extend(f[1])
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
                pad_to_multiple_of=self.data_args.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
                return_token_type_ids=False,
                add_special_tokens=True,
                padding_side=self.data_args.padding_side,
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
                pad_to_multiple_of=self.data_args.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
                return_token_type_ids=False,
                add_special_tokens=True,
                padding_side=self.data_args.padding_side,
            )

            return q_collated, d_collated

        all_query_list_str = [f[0] for f in features]
        all_query_temporal_list_list_str = [f[1] for f in features]

        all_passages_list_list_tuple = [f[2] for f in features]

        all_passages_str = []
        all_passages_temporal_list_list_str = []
        # all_passages_temporal_query_type_list_str = []
        # all_passages_allen_relation_list_str = []

        for i in all_passages_list_list_tuple:
            for j in i:
                all_passages_str.append(j[0])
                all_passages_temporal_list_list_str.append(j[1])
                # all_passages_temporal_query_type_list_str.append(j[2])
                # all_passages_allen_relation_list_str.append(j[3])

        # Process the query side
        q_collated_list = self.tokenizer(
            all_query_list_str,
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
            padding_side=self.data_args.padding_side,
        )

        qt_list = [" ".join(x) for xs in all_query_temporal_list_list_str for x in xs]

        qt_collated_list = self.tokenizer(
            qt_list,
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
            padding_side=self.data_args.padding_side,
        )

        dt_list = [" ".join(x) for xs in all_passages_temporal_list_list_str for x in xs]

        # Process the passage side
        d_collated_list = self.tokenizer(
            all_passages_str,
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
            padding_side=self.data_args.padding_side,
        )

        dt_collated_list = self.tokenizer(
            dt_list,
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
            padding_side=self.data_args.padding_side,
        )

        return (
            q_collated_list,
            qt_collated_list,
        ), (
            d_collated_list,
            dt_collated_list,
            # torch.as_tensor(
            #     [
            #         temporal_query_type_class_id[x]
            #         for x in all_passages_temporal_query_type_list_str
            #     ]
            # ),
            # torch.as_tensor(
            #     [
            #         allen_relation_class_id[x]
            #         for x in all_passages_allen_relation_list_str
            #     ]
            # ),
        )

    def get_temporal_token_spans(
        self,
        passage_char_spans_list,
        passage_span_counts_list,
        d_collated,
        num_neg=2,
        temporal_max_length=16,
        is_passage=False,
    ):
        temporal_token_spans = []
        temporal_tokens_input_ids = []
        char_span_cursor = 0

        token_input_ids = []
        for ix, num_spans in enumerate(passage_span_counts_list):
            offsets = d_collated["offset_mapping"][ix]
            input_ids = d_collated["input_ids"][ix]
            token_spans = []

            for _ in range(num_spans):
                char_start, char_end = passage_char_spans_list[char_span_cursor]
                token_indices = [
                    i
                    for i, (s, e) in enumerate(offsets)
                    if s != 0 or e != 0  # skip special tokens
                    if e > char_start and s < char_end
                ]

                if token_indices:
                    token_spans.append((token_indices[0], token_indices[-1]))
                    _token_input_ids = input_ids[token_indices]
                    token_input_ids.append(
                        F.pad(
                            _token_input_ids,
                            (0, temporal_max_length - len(_token_input_ids)),
                            value=0,
                        )
                    )
                # else:
                #     token_input_ids.append(torch.ones(16)*-100)
                char_span_cursor += 1

            temporal_token_spans.append(token_spans)

            if not is_passage:
                temporal_tokens_input_ids.append(token_input_ids)
                token_input_ids = []
            elif is_passage and (ix + 1) % num_neg == 0:
                temporal_tokens_input_ids.append(token_input_ids)
                token_input_ids = []

        return temporal_token_spans, temporal_tokens_input_ids

    def get_temporal_char_spans(
        self, all_passages_str, all_passages_temporal_list_list_str
    ):
        passage_char_spans_list = []  # flat list of (start_char, end_char) per passage
        passage_span_counts_list = (
            []
        )  # how many spans per passage, to reconstruct later
        # Step 1: Collect passages and raw char spans
        for passage, passage_temporal in zip(
            all_passages_str, all_passages_temporal_list_list_str
        ):
            curr_spans = []
            passage = passage.lower()
            for t in passage_temporal:
                start = passage.find(
                    t.lower()
                )  # There are cases where the extracted temporal expressions are in upper case and the passage is in lower case
                if start != -1:
                    curr_spans.append((start, start + len(t)))

            # For debugging
            # if len(curr_spans) == 0:
            #     print(passage, passage_temporal)
            #     st()
            # print(passage[start:start+len(t)])
            # # This is only for reconstruction loss no need to be exactly the same as the batch size
            # if curr_spans:
            passage_char_spans_list.extend(curr_spans)
            passage_span_counts_list.append(len(curr_spans))
        return passage_char_spans_list, passage_span_counts_list
