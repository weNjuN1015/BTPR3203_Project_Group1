
import os
import pandas as pd
from config import OUTPUT_DIR, SENTIMENT_CSV

def save_report(df: pd.DataFrame) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cols = [c for c in df.columns if not c.startswith("_")] + ["_compound","_label"]
    outpath = os.path.join(OUTPUT_DIR, SENTIMENT_CSV)
    df.to_csv(outpath, index=False, columns=cols)
    print(f"[Reporter] Saved report to {outpath}")
    return outpath
