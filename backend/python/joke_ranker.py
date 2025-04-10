import pandas as pd
from sentence_transformers import SentenceTransformer, util

class JokeRanker:
    def __init__(self, data_source):
        """
        Builds the JokeRanker with either a CSV path or a list of joke dictionaries.
        Each joke dictionary should have: 'text', 'category', and 'rating'.
        """
        if isinstance(data_source, str):
            self.jokes = self._read_from_csv(data_source)
        else:
            self.jokes = data_source

        joke_texts = [j['text'] for j in self.jokes]
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.joke_embeddings = self.encoder.encode(joke_texts, convert_to_tensor=True)

    def _read_from_csv(self, filepath):
        """
        Reads joke data from a CSV file. Returns a list of dictionaries.
        CSV should have 'joke', 'category', and 'rating' columns.
        """
        df = pd.read_csv(filepath).fillna("")
        formatted = []
        for _, row in df.iterrows():
            formatted.append({
                'text': row.get('joke', ''),
                'category': row.get('category', ''),
                'rating': row.get('rating', 0)
            })
        return formatted

    def find_best_matches(self, user_input, limit=5):
        """
        Returns the top matching jokes for a given query.
        Adds score bonuses for category relevance and joke rating.
        """
        input_vector = self.encoder.encode(user_input, convert_to_tensor=True)
        raw_scores = util.cos_sim(input_vector, self.joke_embeddings)[0]

        scored_jokes = []
        for idx, raw_score in enumerate(raw_scores):
            joke_entry = self.jokes[idx]
            score = float(raw_score)

            # Mild boost if query and category line up
            if joke_entry['category'].lower() in user_input.lower():
                score += 0.1

            # Slight boost based on rating (scale 0-5)
            score += float(joke_entry.get('rating', 0)) * 0.01

            scored_jokes.append({
                'text': joke_entry['text'],
                'category': joke_entry.get('category', ''),
                'rating': joke_entry.get('rating', 0),
                'score': score
            })

        return sorted(scored_jokes, key=lambda x: x['score'], reverse=True)[:limit]
