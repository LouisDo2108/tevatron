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
    "matryoshka (our)": "MRL (Our)",
    "madaptor (our)": "M-Adaptor (Our)",
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
def plot_methods(args, ax, results: dict, dims_mapping: dict):
    for method, scores in results.items():
        method = method.lower()
        # print(method)
        
        if method in ["lora", "ts-retriever", "tempretriever", "zero-shot"]:
            dim = sorted(scores.keys())[-1]
            ax.plot(dims_mapping[dim], scores[dim], linestyle="None", marker="*", markersize=6, label=method_mapping[method], color=color_mapping[method])
        elif "nq" in method:
            method = method.split("_")[0]
            dim = sorted(scores.keys())[-1]
            if "timeqa" in args.input_path: 
                if method in ["matryoshka (our)", "madaptor (our)"]:
                    ax.plot(dims_mapping[dim]+0.5, scores[dim], marker="+", markersize=4, color=color_mapping[method], label="NQ (Our)" if "NQ" not in label_set else None)
                    label_set.add("NQ (Our)")
                else:
                    ax.plot(dims_mapping[dim]+0.5, scores[dim], marker="x", markersize=4, color=color_mapping[method], label="NQ" if "NQ" not in label_set else None)
                    label_set.add("NQ")
            else:
                ax.plot(dims_mapping[dim]+0.5, scores[dim], marker="x", markersize=4, color=color_mapping[method], label="NQ" if "NQ" not in label_set else None)
                label_set.add("NQ")
        else:
            points = [
                (dims_mapping[d], scores[d])
                for d in sorted(scores.keys())
            ]
            x, y = zip(*points)
            if "timeqa" in args.input_path: 
                if method in ["matryoshka", "madaptor"]:
                    ax.plot(x, y, marker="o", linestyle="--", markersize=3, linewidth=1.5, label=method_mapping[method], color=color_mapping[method])
                else:
                    ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, label=method_mapping[method], color=color_mapping[method])
            else:
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
    parser.add_argument("--metric", type=str, default="nDCG@10")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=100)
    
    DIMS_MAPPING = {
        64: 0,
        128: 1,
        256: 2,
        512: 3,
        768: 4,
    }
    
    if "bgem3" in args.input_path:
        DIMS_MAPPING[1024] = 5

    results = load_results(args.input_path, DIMS_MAPPING)
    plot_methods(args, ax, results, DIMS_MAPPING)
    
    ax.set_xlabel("Embedding dimension", fontsize=7, labelpad=0)
    ax.set_ylabel(args.metric + r" $\uparrow$", fontsize=7, labelpad=0)
    

    if "NQ" in label_set:
        # Handle the NQ legends
        handles, labels = ax.get_legend_handles_labels()
        nq = mlines.Line2D([], [], linestyle="None", color='black', marker="x",
                            markersize=4, label='NQ')
        handles[-1] = nq
        
        if "timeqa" in args.input_path:
            handles.append(
                mlines.Line2D([], [], linestyle="None", color='black', marker="+", markersize=4, label='NQ (Our)')
            )
            ax.legend(handles=handles, fontsize=6, loc="best", handlelength=3)
        else:
            ax.legend(handles=handles, fontsize=6, loc="best")
        
        temp = np.arange(len(DIMS_MAPPING))
        temp = np.append(temp, temp[-1] + 0.5)
        ax.set_xticks(
            temp, 
            labels=[str(d) for d in DIMS_MAPPING.keys()] + ["NQ"]
        )
    else:
        ax.set_xticks(
            np.arange(len(DIMS_MAPPING)),
            labels=[str(d) for d in DIMS_MAPPING.keys()]
        )
        if "timeqa" in args.input_path:
            print('yes')
            ax.legend(fontsize=6, loc="best", handlelength=3)
        else:
            ax.legend(fontsize=6, loc="best")

    ax.tick_params(axis='both', size=6, labelsize=7, pad=0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(str(args.backbone).upper(), fontsize=8, fontweight="bold", pad=2)
    # fig.suptitle(str(args.backbone).upper(), fontsize=8, fontweight="bold")
    
    fig.tight_layout(pad=0.0)
    plt.savefig(args.output_path, format="pdf", bbox_inches="tight", dpi=100)
    pprint(f"Converted {Path(args.input_path)} to {Path(args.output_path)}")

if __name__ == "__main__":
    main()
