from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import argparse
from pdb import set_trace as st
from pprint import pprint

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
    "matryoshka (our)": "royalblue",
    "madaptor (our)": "green",
}

method_mapping = {
    "matryoshka": "MRL",
    "madaptor": "M-Adaptor",
    "matryoshka (our)": "MRL (Our data)",
    "madaptor (our)": "M-Adaptor (Our data)",
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
def plot_methods(args, ax, ax2, results: dict, dims_mapping: dict):
    for method, scores in results.items():
        method = method.lower()
        
        if method in ["lora", "ts-retriever", "tempretriever", "zero-shot"]:
            dim = sorted(scores.keys())[-1]
            ax.plot(dims_mapping[dim], scores[dim], linestyle="None", marker="*", markersize=6, label=method_mapping[method], color=color_mapping[method])
        elif "nq" in method:
            method = method.split("_")[0]
            # print(method)
            dim = sorted(scores.keys())[-1]
            if method in ["matryoshka (our)", "madaptor (our)"]:
                ax2.plot(dims_mapping[dim]+0.5, scores[dim], marker="+", markersize=4, color=color_mapping[method], label="NQ (Our)" if "NQ" not in label_set else None)
                label_set.add("NQ (Our)")
            else:
                ax2.plot(dims_mapping[dim]+0.5, scores[dim], marker="x", markersize=4, color=color_mapping[method], label="NQ" if "NQ" not in label_set else None)
                label_set.add("NQ")
        else:
            points = [
                (dims_mapping[d], scores[d])
                for d in sorted(scores.keys())
            ]
            x, y = zip(*points)
            if method in ["matryoshka", "madaptor"]:
                ax.plot(x, y, marker="o", linestyle="--", markersize=3, linewidth=1.5, label=method_mapping[method], color=color_mapping[method])
            else:
                ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, label=method_mapping[method], color=color_mapping[method])
        label_set.add(method)

# ------------------------
# Main
# ------------------------
def main():
    setup_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="/home/thuy0050/code/tevatron/src/tevatron/louis/plot/figure1/timeqa/result.csv", type=str)
    parser.add_argument("--output_path", default="/home/thuy0050/code/tevatron/src/tevatron/louis/plot/figure1/timeqa/result.pdf", type=str)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax2 = ax.twinx()
    
    DIMS_MAPPING = {
        64: 0,
        128: 1,
        256: 2,
        512: 3,
        768: 4,
    }

    results = load_results(args.input_path, DIMS_MAPPING)
    plot_methods(args, ax, ax2, results, DIMS_MAPPING)
    
    temp = np.arange(len(DIMS_MAPPING))
    temp = np.append(temp, temp[-1] + 0.5)
    ax.set_xticks(
        temp, 
        labels=[str(d) for d in DIMS_MAPPING.keys()] + ["NQ"]
    )
    
    ax.set_xlabel("Embedding dimension", fontsize=7, labelpad=0)
    ax.set_ylabel(r"F1 $\uparrow$", fontsize=7, labelpad=0) 
    ax2.set_ylabel(r"NQ nDCG@10 $\uparrow$", fontsize=7, labelpad=0) 
    # ax2.set_ylim(15, 30)
    ax2.set_ylim(15, 28)
    
    ax.tick_params(axis='both', size=6, labelsize=7, pad=0)
    ax2.tick_params(axis='both', size=6, labelsize=7, pad=0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax2.grid(False)
    
    ax.set_title("RAG Performance on TimeQA w/ Contriever & Qwen3-8B", fontsize=8, fontweight="bold", pad=2)
    
        
    # Handle the NQ legends
    handles, labels = ax.get_legend_handles_labels()
    nq = mlines.Line2D([], [], linestyle="None", color='black', marker="x",
                        markersize=4, label='NQ')
    handles.append(nq)
    # handles.append(
    #     mlines.Line2D([], [], linestyle="None", color='black', marker="+", markersize=4, label='NQ (Our)')
    # )
    # ax.legend(handles=handles, fontsize=6, loc="best", handlelength=3)
    ax.legend(handles=handles, fontsize=6, loc="lower center", handlelength=3)
    
    fig.tight_layout(pad=0.0)
    plt.savefig(args.output_path, format="pdf", bbox_inches="tight", dpi=100)
    pprint(f"Converted {Path(args.input_path)} to {Path(args.output_path)}")

if __name__ == "__main__":
    main()
