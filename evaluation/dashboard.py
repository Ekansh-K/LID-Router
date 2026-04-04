"""
Dashboard: visualization functions for evaluation results.

These run on CPU and generate static images/HTML — safe for LOCAL machine
once you have the results JSON files from Kaggle.
"""
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from src.utils import get_logger

log = get_logger("dashboard")


def plot_confusion_matrix(lang_list: List[str], matrix: np.ndarray,
                          save_path: str = "./results/confusion_matrix.png",
                          title: str = "LID Confusion Matrix"):
    """Plot NxN confusion heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(10, len(lang_list) * 0.5),
                                     max(8, len(lang_list) * 0.4)))

    # Normalize rows to get rates
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    rates = matrix / row_sums

    sns.heatmap(rates, xticklabels=lang_list, yticklabels=lang_list,
                annot=(len(lang_list) <= 30), fmt=".2f",
                cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted Language")
    ax.set_ylabel("True Language")
    ax.set_title(title)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved to {save_path}")


def plot_routing_distribution(routing_dist: Dict[str, int],
                              save_path: str = "./results/routing_distribution.png"):
    """Bar chart of routing mode distribution."""
    import matplotlib.pyplot as plt

    modes = ["A", "B", "C"]
    counts = [routing_dist.get(m, 0) for m in modes]
    total = sum(counts) or 1
    pcts = [c / total * 100 for c in counts]
    labels = [f"Mode {m}\n({c} / {p:.1f}%)" for m, c, p in zip(modes, counts, pcts)]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black")
    ax.set_ylabel("Number of Utterances")
    ax.set_title("Routing Mode Distribution")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Routing distribution saved to {save_path}")


def plot_cer_comparison(results_dict: Dict[str, float],
                        save_path: str = "./results/cer_comparison.png",
                        title: str = "CER Comparison: System vs Baselines"):
    """Bar chart comparing CER across system variants."""
    import matplotlib.pyplot as plt

    names = list(results_dict.keys())
    values = list(results_dict.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, values, color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    ax.set_xlabel("Character Error Rate (CER)")
    ax.set_title(title)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", ha='left', va='center')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"CER comparison saved to {save_path}")


def plot_per_language_cer(per_lang: Dict[str, dict],
                          save_path: str = "./results/per_language_cer.png"):
    """Per-language CER bar chart sorted by error rate."""
    import matplotlib.pyplot as plt

    langs = sorted(per_lang.keys(), key=lambda l: per_lang[l].get("mean_cer", 0))
    cer_vals = [per_lang[l].get("mean_cer", 0) for l in langs]

    fig, ax = plt.subplots(figsize=(12, max(6, len(langs) * 0.3)))
    colors = ["#2ecc71" if c < 0.2 else "#f39c12" if c < 0.5 else "#e74c3c"
              for c in cer_vals]
    ax.barh(langs, cer_vals, color=colors)
    ax.set_xlabel("Mean CER")
    ax.set_title("Per-Language Character Error Rate")
    ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='0.2 threshold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Per-language CER saved to {save_path}")


def generate_full_dashboard(results_json_path: str,
                            output_dir: str = "./results/dashboard"):
    """Generate all visualizations from a saved results JSON file.
    
    Run this LOCALLY after downloading results from Kaggle.
    """
    with open(results_json_path) as f:
        data = json.load(f)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Routing distribution
    if "routing_distribution" in data:
        plot_routing_distribution(
            data["routing_distribution"],
            f"{output_dir}/routing_distribution.png"
        )

    # Per-language CER
    if "per_language" in data:
        plot_per_language_cer(
            data["per_language"],
            f"{output_dir}/per_language_cer.png"
        )

    log.info(f"Dashboard generated in {output_dir}")
