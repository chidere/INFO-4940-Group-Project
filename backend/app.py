import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from python.query_processing import QueryProcessor
from python.joke_ranker import JokeRanker
from python.text_utils import preprocess
import plotly.graph_objects as go

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

        ranked_jokes = joke_ranker.rank_jokes(search_query, 5, return_latent=True)
        filtered_results = []

        for joke_text, score, latent_vec in ranked_jokes:
            if joke_text in joke_data_map:
                joke_info = joke_data_map[joke_text]
                joke_info["score"] = score
                joke_info["latent_dimensions"] = latent_vec.tolist()[:10]  # top 10 weights
                filtered_results.append(joke_info)

        return filtered_results
    except Exception as e:
        print("Error during joke search:", str(e))
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
                "score": joke.get('score', 1.0),
                "latent_dimensions": joke.get("latent_dimensions", [])  # ✅ Now included
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
        print("Top 5 jokes:", [j for j, _ in ranked])

        def loose_match(query_joke, candidate_joke):
            q_words = set(preprocess(query_joke).split())
            c_words = set(preprocess(candidate_joke).split())
            return len(q_words & c_words) >= max(1, len(q_words) // 2)

        joke_index = next(
            (i for i, (text, _) in enumerate(ranked)
             if loose_match(joke_text, text)),
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
        top_values = [normalized[i] for i in top_indices]

        feature_names = joke_ranker.vectorizer.get_feature_names_out()
        component_words = []
        for i in top_indices:
            top_word_indices = np.argsort(joke_ranker.reducer.components_[i])[-1:]
            label = feature_names[top_word_indices[0]]
            component_words.append(label)

        if not top_values or not component_words:
            return jsonify({
                "data": [],
                "layout": {
                    "title": "No relevant features found for this joke.",
                    "annotations": [{
                        "text": "No relevance found",
                        "showarrow": False,
                        "font": { "size": 16 },
                        "xref": "paper", "yref": "paper",
                        "x": 0.5, "y": 0.5,
                        "xanchor": "center", "yanchor": "middle"
                    }]
                }
            })

        fig = go.Figure(data=go.Scatterpolar(
            r=top_values + [top_values[0]],
            theta=component_words + [component_words[0]],
            fill='toself'
        ))
        fig.update_layout(
            title="Top Contributing Features",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(top_values) * 1.2])
            ),
            showlegend=False
        )

        return jsonify(fig.to_dict())

    except Exception as e:
        print("Explanation error:", e)
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
