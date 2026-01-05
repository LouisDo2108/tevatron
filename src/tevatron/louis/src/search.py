import os
import glob
import logging
import pickle
from argparse import ArgumentParser
from itertools import chain

import faiss
import numpy as np
from tqdm import tqdm
import pandas as pd

from tevatron.retriever.searcher import FaissFlatSearcher
import torch.nn.functional as F
from numpy import linalg as LA

from pdb import set_trace as st

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def search_queries(retriever, q_reps, p_lookup, depth, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, depth, args.batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def search_queries_funnel(retriever, q_reps, p_lookup, depth, args, temporal=False, semantic=False):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, depth, args.batch_size, args.quiet, semantic=semantic, temporal=temporal)
    else:
        all_scores, all_indices = retriever.search(q_reps, depth, semantic=semantic, temporal=temporal)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


class FaissFlatFunnelSearcher:
    def __init__(self, init_reps: np.ndarray):
        self.semantic_dim = 512 # Hard-coded for now
        self.temporal_dim = init_reps.shape[1] - self.semantic_dim
        print(f"Semantic dim {self.semantic_dim}")
        print(f"Temporal dim {self.temporal_dim}")
        
        self.semantic_index = faiss.IndexFlatIP(self.semantic_dim)
        self.temporal_index = faiss.IndexFlatIP(self.temporal_dim)
        self.index = faiss.IndexFlatIP(init_reps.shape[1])

    def add(self, p_reps: np.ndarray, temporal=False, semantic=False):
        # assert temporal ^ semantic, "Only temporal or Only semantic"
        if semantic:
            return self.semantic_index.add(p_reps)
        if temporal:
            return self.temporal_index.add(p_reps)
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int, temporal=False, semantic=False):
        # assert temporal ^ semantic, "Only temporal or Only semantic"
        if semantic:
            return self.semantic_index.search(q_reps, k)
        if temporal:
            return self.temporal_index.search(q_reps, k)
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False, temporal=False, semantic=False):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k, temporal=temporal, semantic=semantic)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices


def run_temporal_retrieval(retriever, p_reps_0, q_reps, look_up, q_lookup, res, co, temporal_k, args, root_dir):
    logger.info('Building Temporal Index')
    p_reps_temporal = p_reps_0[:, 512:] / LA.norm(p_reps_0[:, 512:], axis=1, keepdims=True)
    retriever.add(p_reps_temporal, temporal=True)
    retriever.temporal_index = faiss.index_cpu_to_gpu(res, 0, retriever.temporal_index, co)

    logger.info('Temporal Index Search Start')
    q_reps_temporal = q_reps[:, 512:] / LA.norm(q_reps[:, 512:], axis=1, keepdims=True)
    temporal_all_scores, temporal_psg_indices = search_queries_funnel(
        retriever, q_reps_temporal, look_up, depth=temporal_k, args=args, temporal=True
    )
    logger.info('Temporal Index Search Finished')

    write_ranking(temporal_psg_indices, temporal_all_scores, q_lookup, os.path.join(root_dir, "temporal_rank.txt"))
    
    print(f"After temporal filtering @ {temporal_k}, num passages left: {len(np.unique(temporal_psg_indices))}")

    return temporal_psg_indices, temporal_all_scores


def run_semantic_retrieval(retriever, p_reps_0, q_reps, temporal_psg_indices, q_lookup, res, co, semantic_k, args, root_dir):
    logger.info('Building Semantic Index')
    unique_indices = np.unique(temporal_psg_indices.astype(np.int32))
    p_reps_semantic = p_reps_0[unique_indices, :512] / LA.norm(p_reps_0[unique_indices, :512], axis=1, keepdims=True)
    retriever.add(p_reps_semantic, semantic=True)
    retriever.semantic_index = faiss.index_cpu_to_gpu(res, 0, retriever.semantic_index, co)

    logger.info('Semantic Index Search Start')
    q_reps_semantic = q_reps[:, :512] / LA.norm(q_reps[:, :512], axis=1, keepdims=True)
    semantic_all_scores, semantic_psg_indices = search_queries_funnel(
        retriever, q_reps_semantic, unique_indices, depth=semantic_k, args=args, semantic=True
    )
    logger.info('Semantic Index Search Finished')

    write_ranking(semantic_psg_indices, semantic_all_scores, q_lookup, os.path.join(root_dir, "semantic_rank.txt"))
    
    semantic_psg_indices = semantic_psg_indices.astype(np.int32)
    print(f"After semantic filtering @ {semantic_k}, num passages left: {len(np.unique(semantic_psg_indices))}")

    return semantic_psg_indices, semantic_all_scores


def load_faiss_index_to_device(retriever):
    num_gpus = faiss.get_num_gpus()
    if num_gpus == 0:
        logger.info("No GPU found or using faiss-cpu. Back to CPU.")
    else:
        logger.info(f"Using {num_gpus} GPU")
        if num_gpus == 1:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            res = faiss.StandardGpuResources()
            retriever.index = faiss.index_cpu_to_gpu(res, 0, retriever.index, co)
        else:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            retriever.index = faiss.index_cpu_to_all_gpus(retriever.index, co,
                                                        ngpu=num_gpus)



def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_text', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--funnel', action='store_true')
    temporal_k = 10
    semantic_k = 10

    args = parser.parse_args()

    index_files = glob.glob(args.passage_reps)
    logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

    p_reps_0, p_lookup_0 = pickle_load(index_files[0])
    
    if args.funnel:
        
        root_dir = "/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/ts-retriever/bge-base-en-v1.5/v4_qt_reconstruction_random"
        
        retriever = FaissFlatFunnelSearcher(p_reps_0)
        
        look_up = p_lookup_0
        q_reps, q_lookup = pickle_load(args.query_reps)
        q_reps = q_reps
        
        num_gpus = faiss.get_num_gpus()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        res = faiss.StandardGpuResources()
        
        print(f"num passages left: {len(look_up)}")
        
        # Temporal first
        temporal_psg_indices, temporal_psg_scores = run_temporal_retrieval(retriever, p_reps_0, q_reps, look_up, q_lookup, res, co, temporal_k, args, root_dir)

        semantic_psg_indices, semantic_psg_scores = run_semantic_retrieval(retriever, p_reps_0, q_reps, temporal_psg_indices, q_lookup, res, co, semantic_k, args, root_dir)
        
        # Hybrid search

        # temporal_df = pd.read_csv("/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/ts-retriever/bge-base-en-v1.5/temporal_last_layer_lora_with_normalization_4hardnegatives_95semantic_temporal/temporal_rank.txt", delimiter = "\t", names=["qid", "pid", "score"])
        
        # semantic_df = pd.read_csv("/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/ts-retriever/bge-base-en-v1.5/temporal_last_layer_lora_with_normalization_4hardnegatives_95semantic_temporal/semantic_rank.txt", delimiter = "\t", names=["qid", "pid", "score"])
        
        # temporal_df["new_id"] = [str(x) + "." + str(y) for x, y in zip(temporal_df["qid"], temporal_df["pid"])]
        
        # semantic_df["new_id"] = [str(x) + "." + str(y) for x, y in zip(semantic_df["qid"], semantic_df["pid"])]
        
        # merged = pd.merge(temporal_df, semantic_df, on="new_id", how="right")
        # merged['final_score'] = merged['score_x'] + 2*merged['score_y']
        
        # merged = merged[['qid_y', 'pid_y', 'final_score']].dropna()
        # merged.to_csv("/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/ts-retriever/bge-base-en-v1.5/temporal_last_layer_lora_with_normalization_4hardnegatives_95semantic_temporal/hybrid_rank.txt", index=False, sep='\t', header=False)
        
        # Full

        # retriever.add(p_reps_0[np.unique(semantic_psg_indices)])
        # retriever.index = faiss.index_cpu_to_gpu(res, 0, retriever.index, co)
        
        # logger.info('Index Search Start')
        # all_scores, psg_indices = search_queries(retriever, q_reps, semantic_psg_indices, depth=args.depth, args=args)
        # logger.info('Index Search Finished')
        
        st()
                      
        write_ranking(semantic_psg_indices, semantic_psg_scores, q_lookup, os.path.join(root_dir, "rank.txt"))
        
    else:
        retriever = FaissFlatSearcher(p_reps_0)

        shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
        if len(index_files) > 1:
            shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
        look_up = []
        for p_reps, p_lookup in shards:
            retriever.add(p_reps)
            look_up += p_lookup

        q_reps, q_lookup = pickle_load(args.query_reps)
        q_reps = q_reps

        load_faiss_index_to_device(retriever)

        logger.info('Index Search Start')
        all_scores, psg_indices = search_queries(retriever, q_reps, look_up, depth=args.depth, args=args)
        logger.info('Index Search Finished')

        if args.save_text:
            write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
        else:
            pickle_save((all_scores, psg_indices), args.save_ranking_to)


if __name__ == '__main__':
    main()
