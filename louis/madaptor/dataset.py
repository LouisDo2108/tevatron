import logging

from tevatron.retriever.dataset import TrainDataset
from datasets import load_dataset
from tevatron.retriever.arguments import DataArguments
from pdb import set_trace as st
import random

logger = logging.getLogger(__name__)

class UnsupervisedMAdaptorDataset(TrainDataset):
    def __init__(self, *args, **kwargs):
        super(UnsupervisedMAdaptorDataset, self).__init__(*args, **kwargs)
        self.chunked = False
        if self.train_data is not None:
            self.chunked = "chunkid" in self.train_data[0].keys()

    def __getitem__(self, item):
        content = self.train_data[item]
        content_docid = content["docid"]
        content_chunkid = content.get("chunkid", "")
        content_text = content.get("text", "")

        dummy_query = []
        return dummy_query, (content_docid, content_chunkid, content_text)


class UnsupervisedTemporalMAdaptorDataset(UnsupervisedMAdaptorDataset):
    def __init__(self, *args, **kwargs):
        super(UnsupervisedTemporalMAdaptorDataset, self).__init__(*args, **kwargs)
        self.temporal_dataset = load_dataset(
            "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/tevatron/Tevatron___wikipedia-nq-corpus",
            None,
            data_files="/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/temporal_nobel_prize/train/chunked/corpus_temporal_refined.jsonl",
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            num_proc=self.data_args.num_proc,
        )
        # The temporal dataset should correspond to the training corpus
        assert len(self.temporal_dataset) == len(self.train_data)

    def __getitem__(self, item):
        content = self.train_data[item]
        content_docid = content["docid"]
        content_chunkid = content.get("chunkid", "")
        content_text = content.get("text", "")
        
        temporal = self.temporal_dataset[item]
        content_temporal = temporal.get("temporal")

        dummy_query = []
        return dummy_query, (content_docid, content_chunkid, content_text, content_temporal)


class SupervisedMAdaptorDataset(TrainDataset):

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
        self.data_args = data_args
        self.trainer = trainer

        # Load training data
        self.train_data = load_dataset(
            self.data_args.dataset_name if dataset_name is None else dataset_name,
            self.data_args.dataset_config,
            data_files=(
                self.data_args.dataset_path if dataset_path is None else dataset_path
            ),
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

        # Check if the corpus is chunked
        self.chunked = False
        if self.corpus is not None:
            self.chunked = 'chunkid' in self.corpus[0].keys()

        self.docid_to_index = {}
        if self.chunked and self.corpus is not None:
            for item in self.corpus:
                chunkid = item.get("chunkid", "")
                docid = item.get("docid", "")
                text = item.get("text", "")

                _docid = self.docid_to_index.get(docid, "")
                if _docid == "":
                    self.docid_to_index[docid] = [{chunkid: text}]
                else:
                    self.docid_to_index[docid].append({chunkid: text})
        elif not self.chunked and self.corpus is not None:
            for item in self.corpus:
                docid = item.get("docid", "")
                text = item.get("text", "")
                self.docid_to_index[docid] = text

    def __getitem__(self, item):
        group = self.train_data[item]

        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        query_text = group["query"]
        query_image = query_video = query_audio = None
        formatted_query = (
            self.data_args.query_prefix + query_text,
            query_image,
            query_video,
            query_audio,
        )

        formatted_documents = []
        corpus_documents = []
        # Select positive document
        selected_positive = group["positive_passages"][
            (_hashed_seed + epoch) % len(group["positive_passages"])
        ]
        positive_text = (
            selected_positive["title"] + " " + selected_positive["text"]
            if "title" in selected_positive
            else selected_positive["text"]
        )
        formatted_documents.append(
            (self.data_args.passage_prefix + positive_text, None, None, None)
        )

        """
        The positive passage here refers to the "temporal question" that is related to the document. Let's stick with the name positive query.
        Get the docid corresponds to the positive query.
        Assuming that there is only one positive docid.
        """

        docid = selected_positive["docid"]
        # Retrieve the document(s) associated with the given docid
        chunked_documents = self.docid_to_index[docid]

        if self.chunked:
            # If documents are chunked, iterate through the list of chunked dicts
            # Each dict has a single key-value pair; extract and add to corpus_documents
            corpus_documents.extend(
                (docid, chunkid, doc)
                for pair in chunked_documents
                for chunkid, doc in pair.items()
            )
        else:
            # If not chunked, retrieve the single key-value pair from the dict
            chunkid, doc = next(iter(chunked_documents.items()))
            corpus_documents.append((docid, chunkid, doc))

        # Select negative documents
        negative_size = self.data_args.train_group_size - 1
        if len(group["negative_passages"]) < negative_size:
            selected_negatives = random.choices(
                group["negative_passages"], k=negative_size
            )
        elif self.data_args.train_group_size == 1:
            selected_negatives = []
        else:
            offset = epoch * negative_size % len(group["negative_passages"])
            selected_negatives = list(group["negative_passages"])
            random.Random(_hashed_seed).shuffle(selected_negatives)
            selected_negatives = selected_negatives * 2
            selected_negatives = selected_negatives[offset : offset + negative_size]

        for negative in selected_negatives:
            negative_text = (
                negative["title"] + " " + negative["text"]
                if "title" in negative
                else negative["text"]
            )
            formatted_documents.append(
                (self.data_args.passage_prefix + negative_text, None, None, None)
            )
        return formatted_query, formatted_documents, corpus_documents


class NaiveTemporalDataset(TrainDataset):

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
        self.data_args = data_args
        self.trainer = trainer

        # Load training data
        self.train_data = load_dataset(
            self.data_args.dataset_name if dataset_name is None else dataset_name,
            self.data_args.dataset_config,
            data_files=(
                self.data_args.dataset_path if dataset_path is None else dataset_path
            ),
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

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        query_text = group['query']
        query_image = query_video = query_audio = None
        formatted_query = (
            self.data_args.query_prefix + query_text,
            query_image,
            query_video,
            query_audio,
        )

        formatted_documents = []
        corpus_documents = []

        # Select positive document
        selected_positive = group["positive_passages"][
            (_hashed_seed + epoch) % len(group["positive_passages"])
        ]
        positive_text = (
            selected_positive["title"] + " " + selected_positive["text"]
            if "title" in selected_positive
            else selected_positive["text"]
        )
        formatted_documents.append(
            (self.data_args.passage_prefix + positive_text, selected_positive["temporal"])
        )

        """
        The positive passage here refers to the "temporal question" that is related to the document. Let's stick with the name positive query.
        Get the docid corresponds to the positive query.
        Assuming that there is only one positive docid.
        """

        # docid = selected_positive["docid"]
        # # Retrieve the document(s) associated with the given docid
        # chunked_documents = self.docid_to_index[docid]

        # if self.chunked:
        #     # If documents are chunked, iterate through the list of chunked dicts
        #     # Each dict has a single key-value pair; extract and add to corpus_documents
        #     corpus_documents.extend(
        #         (docid, chunkid, doc)
        #         for pair in chunked_documents
        #         for chunkid, doc in pair.items()
        #     )
        # else:
        #     # If not chunked, retrieve the single key-value pair from the dict
        #     chunkid, doc = next(iter(chunked_documents.items()))
        #     corpus_documents.append((docid, chunkid, doc))

        # Select negative documents
        negative_size = self.data_args.train_group_size - 1
        if len(group["negative_passages"]) < negative_size:
            selected_negatives = random.choices(
                group["negative_passages"], k=negative_size
            )
        elif self.data_args.train_group_size == 1:
            selected_negatives = []
        else:
            offset = epoch * negative_size % len(group["negative_passages"])
            selected_negatives = list(group["negative_passages"])
            random.Random(_hashed_seed).shuffle(selected_negatives)
            selected_negatives = selected_negatives * 2
            selected_negatives = selected_negatives[offset : offset + negative_size]

        for negative in selected_negatives:
            negative_text = (
                negative["title"] + " " + negative["text"]
                if "title" in negative
                else negative["text"]
            )
            formatted_documents.append(
                (
                    self.data_args.passage_prefix + negative_text,
                    negative["temporal"],
                )
            )
        return formatted_query, formatted_documents, corpus_documents


class SentenceTransformerStyleNaiveTemporalDataset(SupervisedMAdaptorDataset):

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
        self.data_args = data_args
        self.trainer = trainer

        # Load training data
        self.train_data = load_dataset(
            self.data_args.dataset_name if dataset_name is None else dataset_name,
            self.data_args.dataset_config,
            data_files=(
                self.data_args.dataset_path if dataset_path is None else dataset_path
            ),
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

    def __getitem__(self, item):
        group = self.train_data[item]

        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        formatted_query = self.data_args.query_prefix + query_text

        formatted_documents = []
        corpus_documents = []
        # Select positive document
        selected_positive = group["positive_passages"][
            (_hashed_seed + epoch) % len(group["positive_passages"])
        ]
        positive_text = (
            selected_positive["title"] + " " + selected_positive["text"]
            if "title" in selected_positive
            else selected_positive["text"]
        )
        formatted_documents.append(
            (
                self.data_args.passage_prefix + positive_text,
                tuple(selected_positive["temporal"]),
            )
        )

        """
        The positive passage here refers to the "temporal question" that is related to the document. Let's stick with the name positive query.
        Get the docid corresponds to the positive query.
        Assuming that there is only one positive docid.
        """

        # docid = selected_positive["docid"]
        # # Retrieve the document(s) associated with the given docid
        # chunked_documents = self.docid_to_index[docid]

        # if self.chunked:
        #     # If documents are chunked, iterate through the list of chunked dicts
        #     # Each dict has a single key-value pair; extract and add to corpus_documents
        #     corpus_documents.extend(
        #         (docid, chunkid, doc)
        #         for pair in chunked_documents
        #         for chunkid, doc in pair.items()
        #     )
        # else:
        #     # If not chunked, retrieve the single key-value pair from the dict
        #     chunkid, doc = next(iter(chunked_documents.items()))
        #     corpus_documents.append((docid, chunkid, doc))

        # Select negative documents
        negative_size = self.data_args.train_group_size - 1
        if len(group["negative_passages"]) < negative_size:
            selected_negatives = random.choices(
                group["negative_passages"], k=negative_size
            )
        elif self.data_args.train_group_size == 1:
            selected_negatives = []
        else:
            offset = epoch * negative_size % len(group["negative_passages"])
            selected_negatives = list(group["negative_passages"])
            random.Random(_hashed_seed).shuffle(selected_negatives)
            selected_negatives = selected_negatives * 2
            selected_negatives = selected_negatives[offset : offset + negative_size]

        for negative in selected_negatives:
            negative_text = (
                negative["title"] + " " + negative["text"]
                if "title" in negative
                else negative["text"]
            )
            formatted_documents.append(
                (
                    self.data_args.passage_prefix + negative_text,
                    tuple(negative["temporal"]),
                )
            )
        return formatted_query, tuple(formatted_documents), tuple(corpus_documents)
