import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from python.query_processing import QueryProcessor
from python.joke_ranker import JokeRanker

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_directory, 'dataset.csv')

# Load jokes
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

# Profanity filter words
inappropriate_words = {"fuck", "shit", "bitch", "cunt", "pussy", "nigger", "slut", "whore"}

def joke_search(query, category_override="", sentiment_filter=""):
    try:
        query_info = query_processor.process_query(query)
        print(f"Processed query: {query_info}")

        category = category_override if category_override else query_info['category']
        sentiment = sentiment_filter if sentiment_filter else query_info['sentiment']
        keywords = query_info['keywords']

        filtered_jokes = []

        for _, row in jokes_df.iterrows():
            joke_text = f"{row.get('title', '')} {row.get('body', '')}".lower()
            category_match = (category == 'general' or category.lower() in str(row.get('category', '')).lower())
            sentiment_match = (sentiment == 'neutral' or sentiment in joke_text)
            clean_joke = not any(word in joke_text for word in inappropriate_words)

            if category_match and sentiment_match and clean_joke:
                filtered_jokes.append(row.to_dict())

        # Use top N or fallback to ranked jokes
        if len(filtered_jokes) < 5:
            search_query = ' '.join(keywords)
            print(f"Searching with fallback: {search_query}")
            ranked_jokes = joke_ranker.rank_jokes(search_query, 10)

            results = []
            for joke_text, score in ranked_jokes:
                joke_data = joke_data_map.get(joke_text)
                if not joke_data:
                    continue

                text = f"{joke_data.get('title', '')} {joke_data.get('body', '')}".lower()
                if category.lower() in str(joke_data.get('category', '')).lower() and \
                   (sentiment == 'neutral' or sentiment in text) and \
                   not any(word in text for word in inappropriate_words):
                    results.append({
                        'title': joke_data.get('title', ''),
                        'body': joke_data.get('body', ''),
                        'category': joke_data.get('category', ''),
                        'score': float(score)
                    })

                if len(results) >= 5:
                    break
            return results
        else:
            for joke in filtered_jokes:
                joke['score'] = 1.0
            return filtered_jokes[:5]
    except Exception as e:
        print(f"Error in joke_search: {str(e)}")
        return []

@app.route("/")
def home():
    return render_template('base.html', title="Joke Recommender")

@app.route("/roast-it")
def search_jokes():
    query = request.args.get("query", "")
    sentiment = request.args.get("sentiment", "").lower()
    category_override = request.args.get("category", "").lower()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        jokes = joke_search(query, category_override, sentiment)
        joke_texts = [
            f"{j['title']}: {j['body']}" if j.get('title') and j.get('body') else j.get('body', '')
            for j in jokes
        ]
        return jsonify({ "jokes": joke_texts })
    except Exception as e:
        return jsonify({ "error": str(e), "jokes": [] }), 500

@app.route("/categories")
def get_categories():
    categories = jokes_df['category'].dropna().unique().tolist()
    return jsonify(categories)

@app.route("/joke/random")
def random_joke():
    if len(jokes_df) > 0:
        random_joke = jokes_df.sample(1).iloc[0].to_dict()
        return jsonify(random_joke)
    else:
        return jsonify({"error": "No jokes available"}), 404

@app.route("/debug/jokes")
def debug_jokes():
    return jsonify({
        "total_jokes": len(jokes_df),
        "joke_texts": len(joke_texts),
        "categories": jokes_df['category'].dropna().unique().tolist(),
        "sample_jokes": jokes_df.head(3).to_dict('records') if len(jokes_df) > 0 else []
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
