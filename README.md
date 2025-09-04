<<<<<<< HEAD

# Social Media Sentiment Analysis (Modular Python Project)

**Dataset detected:** Amazon Fine Food Reviews (Reviews.csv). We treat `Text` as the post and `Time` (epoch) as the timestamp.

## How to run
```bash
pip install -r requirements.txt
python main.py --input Reviews.csv
```
Outputs (CSV + PNGs) will be placed in `outputs/`.

## Files
- `config.py` — tweak column names, date grain, etc.
- `data_loader.py` — reads CSV/JSON, validates columns, parses timestamps.
- `utils.py` — text cleaning, tokenization, simple stopwords, keyword counter.
- `sentiment_analyzer.py` — VADER sentiment scoring + labels.
- `visualizer.py` — 3 plots: distribution, trend, top keywords.
- `reporter.py` — saves a clean `sentiment_report.csv` (original cols + sentiment).
- `main.py` — orchestration + CLI.

## Notes
- No external downloads (e.g., NLTK corpora) are required; `vaderSentiment` includes its own lexicon.
- If your dataset uses different column names, update `config.py` accordingly.
=======
# BTPR3203_Project_Group1
>>>>>>> 566e5d8f7f60ad2724aa7f5c16630dc2545c4e30
