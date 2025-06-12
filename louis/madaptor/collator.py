import logging
from dataclasses import dataclass

from tevatron.retriever.collator import EncodeCollator

logger = logging.getLogger(__name__)


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

        content_ids = [x[0] for x in d]
        texts = [x[1] for x in d]
        max_length = (
            self.data_args.query_max_len
            if self.data_args.encode_is_query
            else self.data_args.passage_max_len
        )
        collated_inputs = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=(
                max_length - 1 if self.data_args.append_eos_token else max_length
            ),
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        if self.data_args.append_eos_token:
            collated_inputs["input_ids"] = [
                x + [self.tokenizer.eos_token_id] for x in collated_inputs["input_ids"]
            ]
        collated_inputs = self.tokenizer.pad(
            collated_inputs,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return query, (content_ids, collated_inputs)
