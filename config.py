
"""
Project configuration.
Adjust these values to match your dataset.
"""
# Input data
INPUT_PATH = "Reviews.csv"  # default; can be overridden by CLI arg in main.py
ID_COLUMN = "Id"
TEXT_COLUMN = "Text"
TIME_COLUMN = "Time"  # epoch seconds in this dataset
SUMMARY_COLUMN = "Summary"  # optional

# Time parsing
TIME_IS_EPOCH = True  # set False if TIME_COLUMN is already a datetime-like string

# Output
OUTPUT_DIR = "outputs"
SENTIMENT_CSV = "sentiment_report.csv"

# Visualization
DATE_GRAIN = "W"  # 'D' daily, 'W' weekly, 'M' monthly
TOP_N_KEYWORDS = 20

# Preprocessing
MIN_TEXT_LEN = 3
LOWERCASE = True
REMOVE_PUNCT = True
REMOVE_NUMBERS = True
STRIP_URLS = True
STRIP_HTML = True
DEDUPLICATE_BY_ID = True

# Random seed
SEED = 42
