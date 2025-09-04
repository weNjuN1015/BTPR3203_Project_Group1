
import os
import pandas as pd
import matplotlib.pyplot as plt
from config import OUTPUT_DIR, DATE_GRAIN, TOP_N_KEYWORDS, TEXT_COLUMN

from utils import STOPWORDS

def _ensure_outdir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_sentiment_distribution(df: pd.DataFrame, save_as="sentiment_distribution.png"):
    _ensure_outdir()
    ax = df["_label"].value_counts().reindex(["positive","neutral","negative"]).plot(kind="bar")
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    fig = ax.get_figure()
    outpath = os.path.join(OUTPUT_DIR, save_as)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[Visualizer] Saved {outpath}")
    return outpath

def plot_sentiment_pie(df, save_as="sentiment_pie.png"):
    _ensure_outdir()
    counts = df["_label"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["green", "gray", "red"])
    ax.set_title("Sentiment Proportion")
    outpath = os.path.join(OUTPUT_DIR, save_as)
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[Visualizer] Saved {outpath}")
    return outpath


def plot_sentiment_trend(df: pd.DataFrame, save_as="sentiment_trend.png"):
    _ensure_outdir()
    ts = df.set_index("_timestamp").resample(DATE_GRAIN)["_compound"].mean()
    ax = ts.plot(kind="line", marker="o")
    ax.set_title(f"Average Sentiment Over Time ({DATE_GRAIN})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Compound Sentiment")
    fig = ax.get_figure()
    outpath = os.path.join(OUTPUT_DIR, save_as)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[Visualizer] Saved {outpath}")
    return outpath

def plot_top_keywords(df: pd.DataFrame, save_as="top_keywords.png", top_n=TOP_N_KEYWORDS):
    _ensure_outdir()
    print(f"[Visualizer] Extracting top {top_n} keywords...")
    if "_clean_text" not in df.columns:
        print("[Visualizer] No '_clean_text' column found. Skipping keywords plot.")
        return None

    # ✅ 优化关键词统计 + 去掉停用词
    tokens = df["_clean_text"].str.split().explode()
    tokens = tokens[tokens.str.len() > 2]
    tokens = tokens[~tokens.isin(STOPWORDS)]  # ✅ 移除停用词
    keyword_counts = tokens.value_counts().head(top_n)

    if keyword_counts.empty:
        print("[Visualizer] No keywords to plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(keyword_counts)), keyword_counts.values)
    ax.set_yticks(range(len(keyword_counts)))
    ax.set_yticklabels(keyword_counts.index)
    ax.invert_yaxis()
    ax.set_title(f"Top {len(keyword_counts)} Keywords (Stopwords removed)")
    ax.set_xlabel("Frequency")
    fig.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, save_as)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[Visualizer] Saved {outpath}")
    return outpath