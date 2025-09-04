
from dataclasses import dataclass
from typing import Literal
import pandas as pd
from config import TEXT_COLUMN, MIN_TEXT_LEN, LOWERCASE, REMOVE_PUNCT, REMOVE_NUMBERS, STRIP_URLS, STRIP_HTML
from utils import clean_text
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception as e:
    raise ImportError("Please install vaderSentiment: pip install vaderSentiment") from e

@dataclass
class SentimentConfig:
    pos_threshold: float = 0.05
    neg_threshold: float = -0.05

class SentimentAnalyzer:
    def __init__(self, config: SentimentConfig | None = None):
        self.config = config or SentimentConfig()
        self.analyzer = SentimentIntensityAnalyzer()

    def score_text(self, text: str) -> dict:
        return self.analyzer.polarity_scores(text)

    def label_from_compound(self, c: float) -> Literal["positive","neutral","negative"]:
        if c >= self.config.pos_threshold:
            return "positive"
        elif c <= self.config.neg_threshold:
            return "negative"
        else:
            return "neutral"

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clean text
        cleaned = df[TEXT_COLUMN].fillna("").astype(str).apply(
            lambda s: clean_text(s, LOWERCASE, REMOVE_PUNCT, REMOVE_NUMBERS, STRIP_URLS, STRIP_HTML)
        )
        lengths = cleaned.str.len()
        mask = lengths >= MIN_TEXT_LEN
        dropped = (~mask).sum()
        if dropped:
            print(f"[SentimentAnalyzer] Dropping {dropped} rows with very short text (< {MIN_TEXT_LEN} chars).")
        df = df.loc[mask].copy()
        df["_clean_text"] = cleaned.loc[mask].values

        scores = df["_clean_text"].apply(self.score_text).apply(pd.Series)
        df["_compound"] = scores["compound"]
        df["_neg"] = scores["neg"]
        df["_neu"] = scores["neu"]
        df["_pos"] = scores["pos"]
        df["_label"] = df["_compound"].apply(self.label_from_compound)
        return df
