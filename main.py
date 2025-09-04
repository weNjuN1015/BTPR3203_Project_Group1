import argparse
import os
import pandas as pd
import time
from data_loader import DataLoader
from visualizer import plot_sentiment_distribution, plot_sentiment_trend, plot_top_keywords, plot_sentiment_pie
from reporter import save_report
from config import INPUT_PATH, OUTPUT_DIR, TEXT_COLUMN, MIN_TEXT_LEN, LOWERCASE, REMOVE_PUNCT, REMOVE_NUMBERS, STRIP_URLS, STRIP_HTML
from utils import clean_text
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def parse_args():
    p = argparse.ArgumentParser(description="Social Media Sentiment Analysis for Social Impact")
    p.add_argument("--input", type=str, default=INPUT_PATH, help="Path to CSV/JSON dataset")
    p.add_argument("--outdir", type=str, default=OUTPUT_DIR, help="Output directory")
    p.add_argument("--date-grain", type=str, default=None, help="Override date grain: D/W/M")
    return p.parse_args()

def main():
    start_time = time.time()  # ✅ 开始计时
    args = parse_args()
    if args.outdir and args.outdir != OUTPUT_DIR:
        os.makedirs(args.outdir, exist_ok=True)

    # ✅ Step 1: Load Data
    loader = DataLoader(args.input)
    df = loader.load()
    print(f"[Main] Loaded {len(df)} rows.")

    # ✅ Step 2: Clean text with progress bar
    print("[Main] Cleaning text...")
    tqdm.pandas(desc="Cleaning text")
    df["_clean_text"] = df[TEXT_COLUMN].fillna("").astype(str).progress_apply(
        lambda s: clean_text(s, LOWERCASE, REMOVE_PUNCT, REMOVE_NUMBERS, STRIP_URLS, STRIP_HTML)
    )

    # ✅ Step 3: Drop short text
    mask = df["_clean_text"].str.len() >= MIN_TEXT_LEN
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"[Main] Dropping {dropped} rows with text length < {MIN_TEXT_LEN}.")
    df = df.loc[mask].copy()

    # ✅ Step 4: Sentiment scoring (only run polarity_scores once)
    print("[Main] Scoring sentiment...")
    analyzer_model = SentimentIntensityAnalyzer()
    tqdm.pandas(desc="Scoring sentiment")
    scores = df["_clean_text"].progress_apply(analyzer_model.polarity_scores)
    scores_df = pd.DataFrame(list(scores))

    # ✅ Rename columns to keep compatibility
    scores_df.rename(columns={
        "compound": "_compound",
        "neg": "_neg",
        "neu": "_neu",
        "pos": "_pos"
    }, inplace=True)

    df = pd.concat([df, scores_df], axis=1)

    # ✅ Step 5: Add sentiment label
    def label_from_compound(c: float) -> str:
        if c >= 0.05:
            return "positive"
        elif c <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["_label"] = df["_compound"].apply(label_from_compound)
    print(f"[Main] Scored {len(df)} rows.")

    # ✅ Step 6: Save report
    csv_path = save_report(df)

     # ✅ Step 7: Generate visualizations
    print("[Main] Generating visualizations...")
    dist_path = plot_sentiment_distribution(df)       # 柱状图
    pie_path = plot_sentiment_pie(df)                # 饼图
    trend_path = plot_sentiment_trend(df)            # 趋势图
    keywords_path = plot_top_keywords(df)            # Top N关键词


    # ✅ End time
    end_time = time.time()
    elapsed = end_time - start_time

    print("\n=== Outputs ===")
    print(f"Report CSV: {csv_path}")
    print(f"Sentiment Distribution: {dist_path}")
    print(f"Sentiment Pie: {pie_path}")
    print(f"Sentiment Trend: {trend_path}")
    print(f"Top Keywords: {keywords_path}")
    
    print(f"\n[Main] Total execution time: {elapsed:.2f} seconds")
if __name__ == "__main__":
    main()
