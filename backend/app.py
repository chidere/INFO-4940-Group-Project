import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from python.query_processing import QueryProcessor
from python.joke_ranker import JokeRanker
from python.text_utils import preprocess

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_directory, 'dataset.csv')

try:
    jokes_df = pd.read_csv(dataset_path)
    print(f"Successfully loaded {len(jokes_df)} jokes from dataset")
except Exception as e:
    print(f"Error loading joke dataset: {str(e)}")
    jokes_df = pd.DataFrame(columns=['id', 'title', 'body', 'category'])

app = Flask(__name__)
CORS(app)

query_processor = QueryProcessor()

joke_texts = []
joke_data_map = {}

for idx, row in jokes_df.iterrows():
    joke_text = f"{row.get('title', '')} {row.get('body', '')}".strip()
    joke_texts.append(joke_text)
    joke_data_map[joke_text] = row.to_dict()

joke_ranker = JokeRanker(joke_texts)

def joke_search(query, category=""):
    try:
        query_info = query_processor.process_query(query)
        if category:
            query_info['category'] = category
        search_query = ' '.join(query_info['keywords'])
        ranked_jokes = joke_ranker.rank_jokes(search_query, 5)
        filtered_results = []
        for joke_text, score in ranked_jokes:
            if joke_text in joke_data_map:
                joke_data = joke_data_map[joke_text]
                if category and category.lower() not in joke_data.get('category', '').lower():
                    continue
                filtered_results.append({
                    'title': joke_data.get('title', ''),
                    'body': joke_data.get('body', ''),
                    'category': joke_data.get('category', ''),
                    'score': float(score)
                })
        return filtered_results
    except Exception as e:
        print(f"Error in joke_search: {str(e)}")
        return []

@app.route("/")
def home():
    return render_template('base.html', title="Joke Recommender")

@app.route("/roast-it")
def search_jokes():
    query = request.args.get("query", "")
    category = request.args.get("category", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        jokes = joke_search(query, category)
        jokes_with_scores = []
        for joke in jokes:
            joke_text = ""
            if joke.get('title') and joke.get('body'):
                joke_text = f"{joke['title']}: {joke['body']}".encode('utf-8', errors='replace').decode('utf-8')
            elif joke.get('body'):
                joke_text = joke['body'].encode('utf-8', errors='replace').decode('utf-8')
            jokes_with_scores.append({
                "joke": joke_text,
                "score": joke.get('score', 1.0)
            })
        return jsonify({ "jokes_with_scores": jokes_with_scores })
    except Exception as e:
        print(f"Error in search_jokes: {str(e)}")
        return jsonify({ "error": str(e), "jokes_with_scores": [] }), 500

@app.route("/categories")
def get_categories():
    categories = jokes_df['category'].dropna().unique().tolist()
    return jsonify(categories)

@app.route("/joke/random")
def random_joke():
    if len(jokes_df) > 0:
        return jsonify(jokes_df.sample(1).iloc[0].to_dict())
    else:
        return jsonify({ "error": "No jokes available" }), 404

@app.route("/debug/jokes")
def debug_jokes():
    return jsonify({
        "total_jokes": len(jokes_df),
        "joke_texts": len(joke_texts),
        "categories": jokes_df['category'].dropna().unique().tolist(),
        "sample_jokes": jokes_df.head(3).to_dict('records') if len(jokes_df) > 0 else []
    })

@app.route("/explanation", methods=["POST"])
def explain():
    try:
        data = request.get_json()
        query = data.get("query", "")
        joke_text = data.get("joke_text", "")

        print("=== /explanation ===")
        print("Query:", query)
        print("Joke received:", joke_text)

        query_info = query_processor.process_query(query)
        search_query = ' '.join(query_info['keywords'])

        ranked = joke_ranker.rank_jokes(search_query, top_n=5)
        joke_index = next(
            (i for i, (text, _) in enumerate(ranked)
             if preprocess(joke_text) in preprocess(text)),
            -1
        )
        if joke_index == -1:
            print("Joke not found in ranked list")
            return jsonify({ "error": "Joke not found in ranked list." }), 400

        true_index = joke_ranker.jokes.index(ranked[joke_index][0])
        cleaned_query = preprocess(query)
        query_vec = joke_ranker.vectorizer.transform([cleaned_query])
        query_reduced = joke_ranker.reducer.transform(query_vec)[0]
        joke_reduced = joke_ranker.joke_reduced[true_index]

        relevance = query_reduced * joke_reduced
        total = np.sum(relevance)
        normalized = relevance / total if total > 0 else np.zeros_like(relevance)

        top_indices = np.argsort(normalized)[-10:][::-1]
        feature_names = joke_ranker.vectorizer.get_feature_names_out()

        keyword_weights = []
        for i in top_indices:
            top_word_index = np.argsort(joke_ranker.reducer.components_[i])[-1:]
            keyword = feature_names[top_word_index[0]]
            keyword_weights.append({
                "word": keyword,
                "weight": round(normalized[i], 4)
            })

        print("Top keywords:", keyword_weights)

        return jsonify({ "keywords": keyword_weights })

    except Exception as e:
        print("Explanation error:", e)
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
