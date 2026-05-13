import os
import glob
import logging
import pickle
from argparse import ArgumentParser
from itertools import chain

import faiss
import numpy as np
from tqdm import tqdm
from time import perf_counter
from pdb import set_trace as st

from tevatron.retriever.searcher import FaissFlatSearcher

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    # psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = [[p_lookup[x] for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices, dtype=np.int32)
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


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_text', action='store_true')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    index_files = glob.glob(args.passage_reps)
    logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')
    
    matryoshka_dim_list = [64, 128, 256, 512, 768]
    retriever_dict = {}
    query_reps_dict = {}
    
    for m in [256, 768]: #matryoshka_dim_list:
        logger.info(f'Loading index for dimension {m}...')
        # index_files = f"/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/tmrl/facebook/contriever/t_64_alpha_0.1/corpus_emb_{m}.pkl"
        index_files = f"/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/time_sensitive_qa/tmrl/facebook/contriever/t_128_alpha_0.1/corpus_emb_{m}.pkl"
        
        
        if m not in retriever_dict:
            p_reps_0, p_lookup_0 = pickle_load(index_files)
            # if m == 768:
            #     p_reps_0 = np.tile(p_reps_0, (10, 1))
            #     p_lookup_0 = list(range(len(p_reps_0))) # np.tile(p_lookup_0, (10, 1)).tolist()
            retriever_dict[m] = FaissFlatSearcher(p_reps_0)
            print("p_reps_0 shape:", p_reps_0.shape)

            # shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
            # if len(index_files) > 1:
            #     shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
            look_up = []
            # for p_reps, p_lookup in shards:
            retriever_dict[m].add(p_reps_0)
            look_up += p_lookup_0

            # q_reps, q_lookup = pickle_load(f"/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/tmrl/facebook/contriever/t_64_alpha_0.1/queries_emb_{m}.pkl")
            q_reps, q_lookup = pickle_load(f"/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/time_sensitive_qa/tmrl/facebook/contriever/t_128_alpha_0.1/queries_emb_{m}.pkl")
            print("q_reps shape:", q_reps.shape)

            query_reps_dict[m] = (q_reps, q_lookup)
            
            retriever_dict[m].save_index(
                os.path.join(os.path.dirname(args.passage_reps), f"{p_reps_0.shape[1]}.faiss_index")
            )
            
            move_index_to_gpu(retriever_dict[m])

    print("FUNNEL SEARCH")
    total_time = 0.0
    for i in range(1):
        args.depth = 100
        m = 256
        start = perf_counter()
        all_scores, psg_indices = search_queries(retriever_dict[m], query_reps_dict[m][0], look_up, args)
        end = perf_counter()
        total_time += (end-start)
        
        unique_indices = np.unique(psg_indices)
        new_lookup = {idx: i for i, idx in enumerate(unique_indices)}
        
        m = 768
        args.depth = 10
        new_p_reps = p_reps_0[unique_indices]
        print("New index shape", new_p_reps.shape)
        new_retriever = FaissFlatSearcher(p_reps_0[unique_indices])
        new_retriever.add(new_p_reps)
        
        start = perf_counter()
        new_all_scores, new_psg_indices = search_queries(new_retriever, query_reps_dict[m][0], list(new_lookup.keys()), args)
        end = perf_counter()
        total_time += (end-start)

        k = np.array(list(new_lookup.keys()))
        mapping_ar = np.zeros(k.max()+1,dtype=unique_indices.dtype) #k,v from approach #1
        mapping_ar[k] = unique_indices
        out = mapping_ar[new_psg_indices]
        
        write_ranking(out, new_all_scores, q_lookup, args.save_ranking_to)
    
    # m = 256
    # args.depth = 100
    # start = perf_counter()
    # for i in range(100):
    #     search_queries(retriever_dict[m], query_reps_dict[m][0], look_up, args)
    # end = perf_counter()
    # print((end - start) / 100)
    
    # m = 768
    # args.depth = 10
    # start = perf_counter()
    # for i in range(100):
    #     new_all_scores, new_psg_indices = search_queries(new_retriever, query_reps_dict[m][0], list(new_lookup.keys()), args)
    # end = perf_counter()
    # print((end - start) / 100)
    
    # Normal search
    # k = 100
    # for m in matryoshka_dim_list:            
    #     print("Matryoshka dimension:", m)
    #     print("Average retrieval time")
    #     start = perf_counter()
    #     for i in range(k):
    #         all_scores, psg_indices = search_queries(retriever_dict[m], query_reps_dict[m][0], look_up, args)
    #     end = perf_counter()
    #     print((end - start) / k)
    #     print()

    # all_scores, psg_indices = search_queries(retriever_dict[m], query_reps_dict[m][0], look_up, args)
    # write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
    
    # if args.save_text:
    #     write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
    # else:
    #     pickle_save((all_scores, psg_indices), args.save_ranking_to)

    # logger.info('Index Search Start')
    # all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)
    # logger.info('Index Search Finished')

def move_index_to_gpu(retriever):
    try:
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
    except Exception as e:
        logger.warning(f"Faiss GPU loading failed, the error is {e}.\nBack to CPU..")


if __name__ == '__main__':
    main()
