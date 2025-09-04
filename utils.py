import re
import html
from typing import List
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Simple English stopwords list
STOPWORDS = set(
    "a an the and or but if then else when at by for with about against between into through during before after above below to from up down in out on off over under again further"
    " is are was were be been being have has had do does did doing would should could can must might may will having not no nor only own same so than too very s t don should ve "
    "i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who "
    "whom this that these those am as of each few more most other some such too very can will just don should now".split())

# Clean text by removing HTML, URLs, punctuation, numbers, etc.
def clean_text(text: str, lowercase=True, remove_punct=True, remove_numbers=True, strip_urls=True, strip_html=True) -> str:
    if not isinstance(text, str):
        return ""
    s = text
    if strip_html:
        s = re.sub(r"<[^>]+>", " ", s)  # Remove HTML tags
        s = html.unescape(s)  # Convert HTML entities
    if strip_urls:
        s = re.sub(r"http[s]?://\S+|www\.\S+", " ", s)  # Remove URLs
    if lowercase:
        s = s.lower()
    if remove_punct:
        s = re.sub(r"[^\w\s]", " ", s)  # Remove punctuation
    if remove_numbers:
        s = re.sub(r"\d+", " ", s)  # Remove numbers
    s = re.sub(r"\s+", " ", s).strip()  # Remove extra whitespace
    return s

# Simple whitespace tokenization
def tokenize(text: str) -> List[str]:
    return [t for t in text.split() if t]

# Remove stopwords and very short tokens
def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

# Keep only tokens with specified part-of-speech tags
def filter_tokens_by_pos(tokens: List[str], allowed_pos=("NOUN", "VERB")) -> List[str]:
    """
    Keep tokens of specific POS.
    Default: nouns and verbs only.
    """
    doc = nlp(" ".join(tokens))
    return [token.text for token in doc if token.pos_ in allowed_pos]

# Get top N frequent terms from a Series of texts
def top_n_terms(texts, n=20):
    all_tokens = texts.str.split().explode()  # Flatten all tokens
    all_tokens = all_tokens[all_tokens.str.len() > 2]  # Remove very short tokens
    return all_tokens.value_counts().head(n).items()  # Return top N terms
