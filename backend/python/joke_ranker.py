import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from python.svd_reducer import SVDReducer
from text_utils import preprocess

class JokeRanker:
    def __init__(self, joke_data, n_components=100):
        """Initialize the joke ranker with jokes data and SVD support."""
        if isinstance(joke_data, str):
            self.jokes = self.load_jokes_from_file(joke_data)
        else:
            self.jokes = joke_data

       
        self.jokes_cleaned = [preprocess(joke) for joke in self.jokes]

    
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.joke_vectors = self.vectorizer.fit_transform(self.jokes_cleaned)

        # Apply SVD
        self.reducer = SVDReducer(n_components=n_components)
        self.joke_reduced = self.reducer.fit(self.joke_vectors)

    def load_jokes_from_file(self, joke_file):
        """Load jokes from a CSV file."""
        df = pd.read_csv(joke_file)
        return df['joke'].fillna("").tolist()

    def rank_jokes(self, query, top_n=5):
        """Rank jokes based on cosine similarity in SVD-reduced space."""
        cleaned_query = preprocess(query)  
        query_vec = self.vectorizer.transform([cleaned_query])
        query_reduced = self.reducer.transform(query_vec)

        ranked_indices, similarities = self.reducer.compute_similarity(
            query_reduced, self.joke_reduced, top_n
        )
        return [(self.jokes[i], similarities[i]) for i in ranked_indices]
