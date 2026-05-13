import argparse
from pathlib import Path
from pdb import set_trace as st
from pprint import pprint

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ------------------------
# Configuration
# ------------------------
color_mapping = {
    "lora": "black",
    "matryoshka": "royalblue",
    "temporal": "orange",
    "madaptor": "green",
    "ts-retriever": "violet",
    "tempretriever": "goldenrod",
    "zero-shot": "tomato",
}

metric_mapping = {
    "ndcg": "nDCG@10",
    "recall100": "Recall@100",
    "nq": "NQ nDCG@10"
}

method_mapping = {
    "contriever": "Contriever",
    "bge": "BGE",
    "matryoshka": "MRL",
    "madaptor": "M-Adaptor",
    "temporal": "TMRL",
    "zero-shot": "0-shot",
    "lora": "LoRA",
    "ts-retriever": "Ts-Retriever",
    "tempretriever": "TempRetriever",
}

# ------------------------
# Plot style
# ------------------------
def setup_style():
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    sns.despine(left=False)


# ------------------------
# Data loading
# ------------------------
def load_results(csv_path: Path, dims_mapping: dict) -> dict:
    """
    Load CSV where each row is a method and columns are dimensions.

    Returns:
        dict[str, dict[int, float]]
    """
    df = pd.read_csv(csv_path, names=["method"] + [str(d) for d in dims_mapping.keys()])

    results = {}
    for _, row in df.iterrows():
        method = row["method"]
        results[method] = {d: row[str(d)] for d in dims_mapping.keys()}

    return results


# ------------------------
# Plotting
# ------------------------
label_set = set()
def plot_methods(ax, results: dict, dims_mapping: dict):
    for method, scores in results.items():
        method = method.lower()
        # print(method)
        
        if method in ["lora", "ts-retriever", "tempretriever", "zero-shot"]:
            dim = sorted(scores.keys())[-1]
            ax.plot(dims_mapping[dim], scores[dim], marker="*", markersize=6, label=method_mapping[method], color=color_mapping[method])
        elif "nq" in method:
            method = method.split("_")[0]
            dim = sorted(scores.keys())[-1]
            ax.plot(dims_mapping[dim]+0.5, scores[dim], marker="x", markersize=6, color=color_mapping[method], label="NQ" if "NQ" not in label_set else None)
            label_set.add("NQ")
        else:
            points = [
                (dims_mapping[d], scores[d])
                for d in sorted(scores.keys())
            ]
            x, y = zip(*points)
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, label=method_mapping[method], color=color_mapping[method])

        label_set.add(method)

# ------------------------
# Main
# ------------------------
def main():
    setup_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(5, 7), dpi=100)
    
    if args.metric != "nq":
        results = pd.read_csv(args.input_path)
        results = results.pivot(index="t", columns="dim")
        # print(results.stack().transpose().transpose())
        
        ax = sns.heatmap(
            data=results.stack().transpose(),
            annot=True,
            linewidth=0.0,
            cmap="viridis",
            center=True,
            fmt='.1f',
            square=True,
            cbar=False,
            annot_kws={"fontsize":6}
        )
        # ax.tick_params(axis='both', size=6, labelsize=7, pad=0)
        # ax.grid(True, linestyle="--", alpha=0.5)
        # ax.legend(fontsize=6, loc="best")
        ax.set_xlabel(r"temporal subspace dimension $t$ -  embedding dimension $m$",labelpad=0, fontsize=7, weight ='bold')
        ax.set_ylabel(r"$\alpha$",labelpad=0, fontsize=7, weight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), weight='bold')
        ax.tick_params(axis='both', labelsize=6, labelrotation=45, pad=0)
        ax.set_title(f"{method_mapping[args.backbone]} {metric_mapping[args.metric]}", fontsize=8, fontweight="bold", pad=2)
    else:
        results = pd.read_csv(args.input_path,index_col="t")
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
        ax = sns.heatmap(
            data=results,
            annot=True,
            linewidth=0.0,
            cmap="viridis",
            center=False,
            fmt='.1f',
            square=True,
            cbar=False,
            vmin=-20,
            vmax=20,
            annot_kws={"fontsize":6}
        )
        # ax.set_xlabel("temporal subspace dimension - full embedding dimension",labelpad=0, fontsize=7)
        # ax.set_ylabel("alpha",labelpad=0, fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), weight='bold')
        ax.tick_params(axis='both', labelsize=6, labelrotation=60, pad=0)
        ax.set_title(f"{method_mapping[args.backbone]} {metric_mapping[args.metric]}", fontsize=8, fontweight="bold", pad=2)

    fig.tight_layout(pad=0.0)
    plt.savefig(args.output_path, format="pdf", bbox_inches="tight", dpi=100)
    pprint(f"Converted {Path(args.input_path)} to {Path(args.output_path)}")

if __name__ == "__main__":
    main()
