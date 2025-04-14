import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from python.query_processing import QueryProcessor
from python.joke_ranker import JokeRanker
from python.joke_retriever import retrieve_jokes  # Import retrieve_jokes here

# ROOT_PATH for linking with all your files. 
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the CSV file relative to the current script
dataset_path = os.path.join(current_directory, 'dataset.csv')

# Load jokes from CSV
try:
    jokes_df = pd.read_csv(dataset_path)
    print(f"Successfully loaded {len(jokes_df)} jokes from dataset")
except Exception as e:
    print(f"Error loading joke dataset: {str(e)}")
    jokes_df = pd.DataFrame(columns=['id', 'title', 'body', 'category'])

app = Flask(__name__)
CORS(app)

query_processor = QueryProcessor()

# Convert the dataframe into a list of dictionaries for easy processing
jokes_data = jokes_df.to_dict('records')

# Initialize JokeRanker with all the joke texts for ranking
joke_texts = [f"{row.get('title', '')} {row.get('body', '')}".strip() for idx, row in jokes_df.iterrows()]
joke_ranker = JokeRanker(joke_texts)

def joke_search(query, category=""):
    try:
        # Step 1: Process the query to extract keywords and category
        query_info = query_processor.process_query(query)
        print(f"Processed query: {query_info}")
        
        # Override category if provided as a parameter
        if category:
            query_info['category'] = category
            
        # Step 2: Retrieve jokes based on category and filter out inappropriate content
        filtered_jokes = retrieve_jokes(query_info['category'], jokes_data)
        print(f"Found {len(filtered_jokes)} jokes after filtering")
        
        # Step 3: Rank jokes based on relevance to the search query (using cosine similarity)
        ranked_jokes = joke_ranker.rank_jokes(query_info['keywords'], 5)
        print(f"Ranked {len(ranked_jokes)} jokes")
        
        # Step 4: Apply category filter if provided
        final_jokes = []
        for joke_text, score in ranked_jokes:
            if joke_text in [joke['title'] + ' ' + joke['body'] for joke in filtered_jokes]:
                joke_data = next(joke for joke in filtered_jokes if joke['title'] + ' ' + joke['body'] == joke_text)
                final_jokes.append({
                    'title': joke_data.get('title', ''),
                    'body': joke_data.get('body', ''),
                    'category': joke_data.get('category', ''),
                    'score': float(score)
                })
        
        return final_jokes
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
        
        if jokes:
            return jsonify({
                "jokes_with_scores": jokes
            })
        else:
            return jsonify({
                "error": "No appropriate jokes found for this query or category"
            }), 404
    except Exception as e:
        print(f"Error in search_jokes: {str(e)}")
        return jsonify({
            "error": str(e),
            "jokes_with_scores": []
        }), 500

@app.route("/categories")
def get_categories():
    """Return all available joke categories"""
    categories = jokes_df['category'].dropna().unique().tolist()
    return jsonify(categories)

@app.route("/joke/random")
def random_joke():
    """Return a random joke"""
    if len(jokes_df) > 0:
        random_joke = jokes_df.sample(1).iloc[0].to_dict()
        return jsonify(random_joke)
    else:
        return jsonify({"error": "No jokes available"}), 404

# Debug endpoint to check if jokes are loaded correctly
@app.route("/debug/jokes")
def debug_jokes():
    """Return information about loaded jokes"""
    return jsonify({
        "total_jokes": len(jokes_df),
        "categories": jokes_df['category'].dropna().unique().tolist(),
        "sample_jokes": jokes_df.head(3).to_dict('records') if len(jokes_df) > 0 else []
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
