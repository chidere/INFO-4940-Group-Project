import pandas as pd
from sentence_transformers import SentenceTransformer, util

class JokeRanker:
    def __init__(self, joke_data):
        """
        Initialize the joke ranker with jokes data.
        Args:
            joke_data: Either a file path to a CSV or a list of joke texts
        """
        if isinstance(joke_data, str):
            self.jokes = self.load_jokes_from_file(joke_data)
        else:
            self.jokes = joke_data

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.joke_embeddings = self.model.encode(self.jokes, convert_to_tensor=True)

    def load_jokes_from_file(self, joke_file):
        """Load jokes from a CSV file."""
        df = pd.read_csv(joke_file)
        return df['joke'].fillna("").tolist()

    def rank_jokes(self, query, top_n=5):
        """
        Rank jokes based on semantic similarity to user input.
        Args:
            query: A string input from the user
            top_n: Number of top jokes to return
        Returns:
            List of tuples: (joke_text, similarity_score)
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self.joke_embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:top_n]
        return [(self.jokes[i], float(similarities[i])) for i in top_indices]
