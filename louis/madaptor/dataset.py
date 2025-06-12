import logging

from tevatron.retriever.dataset import TrainDataset

logger = logging.getLogger(__name__)

class UnsupervisedMAdaptorDataset(TrainDataset):
    def __getitem__(self, item):
        content = self.train_data[item]
        content_id = content["docid"]
        content_text = content.get("text", "")

        dummy_query = []
        return dummy_query, (content_id, content_text)


# class SupervisedMAdaptorDataset(TrainDataset):
#     def __getitem__(self, item):
#         content = self.train_data[item]

#         epoch = int(self.trainer.state.epoch)
#         _hashed_seed = hash(item + self.trainer.args.seed)

#         query_text = group["query"]
#         query_image = query_video = query_audio = None
#         formatted_query = (
#             self.data_args.query_prefix + query_text,
#             query_image,
#             query_video,
#             query_audio,
#         )

#         formatted_documents = []
#         # Select positive document
#         selected_positive = group["positive_passages"][
#             (_hashed_seed + epoch) % len(group["positive_passages"])
#         ]
#         positive_text = (
#             selected_positive["title"] + " " + selected_positive["text"]
#             if "title" in selected_positive
#             else selected_positive["text"]
#         )
#         formatted_documents.append(
#             (self.data_args.passage_prefix + positive_text, None, None, None)
#         )

#         # Select negative documents
#         negative_size = self.data_args.train_group_size - 1
#         if len(group["negative_passages"]) < negative_size:
#             selected_negatives = random.choices(
#                 group["negative_passages"], k=negative_size
#             )
#         elif self.data_args.train_group_size == 1:
#             selected_negatives = []
#         else:
#             offset = epoch * negative_size % len(group["negative_passages"])
#             selected_negatives = list(group["negative_passages"])
#             random.Random(_hashed_seed).shuffle(selected_negatives)
#             selected_negatives = selected_negatives * 2
#             selected_negatives = selected_negatives[offset : offset + negative_size]

#         for negative in selected_negatives:
#             negative_text = (
#                 negative["title"] + " " + negative["text"]
#                 if "title" in negative
#                 else negative["text"]
#             )
#             formatted_documents.append(
#                 (self.data_args.passage_prefix + negative_text, None, None, None)
#             )

#         content_id = content["docid"]
#         content_text = content.get("text", "")

#         dummy_query = []

#         return relevants, formatted_query, formatted_documents
#         return dummy_query, (content_id, content_text)
