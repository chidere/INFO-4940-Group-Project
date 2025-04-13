# text_utils.py
import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()@\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
