import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class JokeRanker:
    def __init__(self, joke_file):
        """Initialize the joke ranker with a joke dataset."""
        self.jokes = self.load_jokes(joke_file)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.joke_vectors = self.vectorizer.fit_transform(self.jokes)

    def load_jokes(self, joke_file):
        """Load jokes from a file (assumes CSV with a 'joke' column)."""
        df = pd.read_csv(joke_file)
        return df['joke'].fillna("").tolist()

    def rank_jokes(self, query, top_n=5):
        """Rank jokes based on cosine similarity to user input."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.joke_vectors).flatten()
        ranked_indices = np.argsort(similarities)[::-1][:top_n]
        return [(self.jokes[i], similarities[i]) for i in ranked_indices]

# Example Usage:
if __name__ == "__main__":
    joke_ranker = JokeRanker("jokes.csv")
    user_query = input("Enter a topic for a joke: ")
    ranked_jokes = joke_ranker.rank_jokes(user_query)

    print("\nBest Matching Jokes:")
    for joke, score in ranked_jokes:
        print(f"- {joke} (Score: {score:.2f})")
