import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from python.text_utils import preprocess  

class QueryProcessor:
    """
    Handles processing of user queries for the joke recommender system.
    Extracts keywords, identifies humor categories, has "Did you mean?" feature, and prepares queries
    for joke retrieval.
    """
    def __init__(self):
        self.humor_categories = {
            'pun': ['pun', 'wordplay', 'play on words'],
            'one-liner': ['one liner', 'one-liner', 'short', 'quick'],
            'blonde': ['blonde', 'dumb', 'silly'],
            'general': ['general', 'normal', 'regular']
        }
        self.joke_subjects = [
            'food', 'work', 'school', 'politics', 'animals', 'sports',
            'relationships', 'technology', 'health', 'money', 'travel'
        ]
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )

    def preprocess_query(self, query):
        """Normalize user query using shared function."""
        return preprocess(query)  

    def extract_keywords(self, query):
        words = query.split()
        keywords = [word for word in words if word in self.joke_subjects]
        if not keywords:
            words.sort(key=len, reverse=True)
            keywords = words[:3] if len(words) >= 3 else words
        return keywords

    def identify_humor_category(self, query):
        query_lower = query.lower()
        for category, keywords in self.humor_categories.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return category
        return 'general'

    def detect_sentiment(self, query):
        query_lower = query.lower()
        sentiment_keywords = {
            'sarcastic': ['sarcastic', 'sarcasm', 'ironic', 'irony'],
            'clever': ['clever', 'witty', 'smart', 'intelligent'],
            'dark': ['dark', 'edgy', 'morbid', 'black humor'],
            'clean': ['clean', 'family friendly', 'pg', 'appropriate']
        }
        for sentiment, keywords in sentiment_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return sentiment
        return 'neutral'

    def get_did_you_mean_suggestions(self, query, joke_corpus, max_suggestions=3):
        processed_query = self.preprocess_query(query)
        query_keywords = self.extract_keywords(processed_query)
        suggestions = []
        for potential_topic in self.joke_subjects:
            for keyword in query_keywords:
                if (keyword in potential_topic or potential_topic in keyword) and potential_topic not in suggestions:
                    suggestions.append(f"jokes about {potential_topic}")
                    break
        return suggestions[:max_suggestions]

    def process_query(self, query):
        processed_query = self.preprocess_query(query)
        keywords = self.extract_keywords(processed_query)
        category = self.identify_humor_category(query)
        sentiment = self.detect_sentiment(query)
        all_category_keywords = [kw for sublist in self.humor_categories.values() for kw in sublist]
        filtered_keywords = [kw for kw in keywords if kw not in all_category_keywords]
        query_info = {
            'original_query': query,
            'processed_query': processed_query,
            'keywords': filtered_keywords if filtered_keywords else keywords,
            'category': category,
            'sentiment': sentiment
        }
        return query_info
