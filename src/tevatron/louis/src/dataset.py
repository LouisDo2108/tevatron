import logging
import random
from pdb import set_trace as st

from datasets import load_dataset

from tevatron.retriever.arguments import DataArguments
from tevatron.retriever.dataset import TrainDataset

from sutime import SUTime

def init_sutime():
    global sutime_instance
    sutime_instance = SUTime(mark_time_ranges=True, include_range=True)
    return sutime_instance


logger = logging.getLogger(__name__)


class TemporalDataset(TrainDataset):

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

        self.passage_prefix = self.data_args.passage_prefix.replace("\\n", "\n").strip()
        if self.passage_prefix != "":
            self.passage_prefix += " "
            
        self.query_prefix = self.data_args.query_prefix.replace("\\n", "\n").strip()
        if self.query_prefix != "":
            self.query_prefix += " "

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        """
        Note that for temporal dataset, since the "query" here is actually the passage and positive/negative "passages" are actually the "query"; 
        *** during training, we should add the passage_prefix to the "query" and query_prefix to the "passages"
        *** during evaluation/inference, they are normal.
        *** This is by swapping query_prefix and passage_prefix in the data_args.
        """

        query_text = group.get('query', "")  # type: ignore
        query_temporal = group.get("temporal", "")  # type: ignore

        formatted_query = (
            self.passage_prefix + query_text
        )  # reversed role of passage_prefix and query_prefix

        formatted_documents = []

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
                self.query_prefix + positive_text, # reversed role of passage_prefix and query_prefix
                selected_positive.get('temporal', ""),
                selected_positive.get('temporal_query_type', ""), 
                selected_positive.get('allen_relation', ""),
            )
        )

        """
        The positive passage here refers to the "temporal question" that is related to the document. Let's stick with the name positive query.
        Get the docid corresponds to the positive query.
        Assuming that there is only one positive docid.
        """

        # Select negative documents
        negative_size = self.data_args.train_group_size - 1
        num_negatives = len(group.get("negative_passages", []))

        # if group.get("negative_passages", None) is None or self.data_args.train_group_size == 1:
        #     selected_negatives = []
        # elif len(group['negative_passages']) < negative_size:
        #     selected_negatives = random.choices(
        #         group['negative_passages'], k=negative_size
        #     )
        if num_negatives > 0 and num_negatives < negative_size:
            selected_negatives = random.choices(
                group["negative_passages"], k=negative_size
            )
        elif num_negatives == 0 or self.data_args.train_group_size == 1:
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
                    self.query_prefix + negative_text,
                    negative.get("temporal", ""),
                    negative.get("temporal_query_type", ""),
                    negative.get("allen_relation", ""),
                )
            )
        return formatted_query, query_temporal, formatted_documents


class TempRetrieverTemporalDataset(TrainDataset):

    def __init__(
        self,
        data_args: DataArguments,
        sutime_parsing=False,
        trainer=None,
        dataset_name=None,
        corpus_name=None,
        dataset_path=None,
        corpus_path=None,
        corpus_assets_path=None,
    ):
        self.data_args = data_args
        self.trainer = trainer
        self.sutime_parsing = sutime_parsing

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

        self.passage_prefix = self.data_args.passage_prefix.replace("\\n", "\n").strip()
        if self.passage_prefix != "":
            self.passage_prefix += " "
            
        self.query_prefix = self.data_args.query_prefix.replace("\\n", "\n").strip()
        if self.query_prefix != "":
            self.query_prefix += " "

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        """
        Note that for temporal dataset, since the "query" here is actually the passage and positive/negative "passages" are actually the "query"; 
        *** during training, we should add the passage_prefix to the "query" and query_prefix to the "passages"
        *** during evaluation/inference, they are normal.
        *** This is by swapping query_prefix and passage_prefix in the data_args.
        """

        query_text = group.get('query', "")  # type: ignore
        query_temporal = group.get("temporal", "")  # type: ignore

        formatted_query = (
            self.passage_prefix + query_text
        )  # reversed role of passage_prefix and query_prefix

        formatted_documents = []

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
                self.query_prefix + positive_text, # reversed role of passage_prefix and query_prefix
                selected_positive.get('temporal', ""),
                selected_positive.get('temporal_query_type', ""), 
                selected_positive.get('allen_relation', ""),
            )
        )

        """
        The positive passage here refers to the "temporal question" that is related to the document. Let's stick with the name positive query.
        Get the docid corresponds to the positive query.
        Assuming that there is only one positive docid.
        """

        # Select negative documents
        negative_size = self.data_args.train_group_size - 1
        
        num_negatives = len(group.get("negative_passages", []))
        
        if num_negatives > 0 and num_negatives < negative_size:
            selected_negatives = random.choices(
                group["negative_passages"], k=negative_size
            )
        elif num_negatives == 0 or self.data_args.train_group_size == 1:
            selected_negatives = []
        else:
            offset = epoch * negative_size % num_negatives
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
                    self.query_prefix + negative_text,
                    negative.get("temporal", ""),
                    negative.get("temporal_query_type", ""),
                    negative.get("allen_relation", ""),
                )
            )
        
        if query_temporal == "" and self.sutime_parsing:
            query_temporal = sutime_instance.parse(query_text)
            query_temporal = ", ".join([t["text"] for t in query_temporal])
        
        return formatted_query, query_temporal, formatted_documents
