import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class JokeRanker:
    def __init__(self, joke_data):
        """Initialize the joke ranker with jokes data.
        Args:
            joke_data: Either a file path to a CSV or a list of joke texts
        """
        if isinstance(joke_data, str):
            self.jokes = self.load_jokes_from_file(joke_data)
        else:
            self.jokes = joke_data
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.joke_vectors = self.vectorizer.fit_transform(self.jokes)
    
    def load_jokes_from_file(self, joke_file):
        """Load jokes from a CSV file."""
        df = pd.read_csv(joke_file)
        return df['joke'].fillna("").tolist()
    
    def rank_jokes(self, query, top_n=5):
        """Rank jokes based on cosine similarity to user input."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.joke_vectors).flatten()
        ranked_indices = np.argsort(similarities)[::-1][:top_n]
        return [(self.jokes[i], similarities[i]) for i in ranked_indices]
