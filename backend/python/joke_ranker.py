import pandas as pd
from sentence_transformers import SentenceTransformer, util

class JokeRanker:
    def __init__(self, joke_data):
        """
        Initialize JokeRanker with joke text, category, and rating.
        Args:
            joke_data: CSV file path or list of dicts with keys 'joke', 'category', and 'rating'.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        if isinstance(joke_data, str):
            self.jokes = self.load_jokes_from_file(joke_data)
        else:
            self.jokes = joke_data

        self.joke_texts = [j['joke'] for j in self.jokes]
        self.joke_vectors = self.model.encode(self.joke_texts, convert_to_tensor=True)

    def load_jokes_from_file(self, filepath):
        df = pd.read_csv(filepath)
        df = df.fillna({'joke': '', 'category': '', 'rating': 0})
        jokes = df.to_dict('records')
        return jokes

    def rank_jokes(self, query, top_n=5, expected_category=None):
        """
        Ranks jokes based on semantic similarity to the query.
        Optionally boosts jokes that match a given category.

        Args:
            query: User input string
            top_n: Number of top jokes to return
            expected_category: (optional) Expected humor category like 'Puns'

        Returns:
            List of (joke_text, final_score, metadata_dict) tuples
        """
        query_vec = self.model.encode(query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_vec, self.joke_vectors)[0]

        results = []
        for i, sim in enumerate(similarities):
            joke = self.jokes[i]
            rating = joke.get('rating', 0)
            category = joke.get('category', '').lower()
            normalized_rating = min(max(rating / 5.0, 0), 1)  # Normalize rating to [0,1]

            # Optional category boost
            category_boost = 0.1 if expected_category and expected_category.lower() in category else 0.0

            # Final score
            final_score = float(sim) * 0.8 + normalized_rating * 0.1 + category_boost

            results.append((joke['joke'], final_score, joke))

        # Sort by final_score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_n]
