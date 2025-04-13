from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class SVDReducer:
    def __init__(self, n_components=100):
        """Initialize the SVD reducer with specified number of components."""
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.fitted = False

    def fit(self, tfidf_matrix):
        """Fit the SVD model to the TF-IDF matrix."""
        self.joke_reduced = self.svd.fit_transform(tfidf_matrix)
        self.fitted = True
        return self.joke_reduced

    def transform(self, tfidf_vector):
        """Transform a new TF-IDF vector into the reduced SVD space."""
        if not self.fitted:
            raise ValueError("SVDReducer must be fit before calling transform().")
        return self.svd.transform(tfidf_vector)

    def compute_similarity(self, query_reduced, corpus_reduced, top_n=5):
        """Compute cosine similarities between reduced query and reduced corpus."""
        similarities = cosine_similarity(query_reduced, corpus_reduced).flatten()
        ranked_indices = similarities.argsort()[::-1][:top_n]
        return ranked_indices, similarities
