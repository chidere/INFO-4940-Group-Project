import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from python.query_processing import QueryProcessor
from python.joke_ranker import JokeRanker

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the csv file relative to the current script
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

joke_texts = []
joke_data_map = {}

for idx, row in jokes_df.iterrows():
    # Create standardized joke text
    joke_text = f"{row.get('title', '')} {row.get('body', '')}".strip()
    joke_texts.append(joke_text)
    joke_data_map[joke_text] = row.to_dict()

joke_ranker = JokeRanker(joke_texts)

def joke_search(query, category=""):
    try:
        # Process the query to extract information
        query_info = query_processor.process_query(query)
        print(f"Processed query: {query_info}")
        
        # Override category if provided as a parameter
        if category:
            query_info['category'] = category
            
        # Get keywords for search
        search_query = ' '.join(query_info['keywords'])
        print(f"Searching with keywords: {search_query}")
        
        # Get ranked jokes using cosine similarity
        ranked_jokes = joke_ranker.rank_jokes(search_query, 5)
        print(f"Found {len(ranked_jokes)} ranked jokes")
        
        # If a category filter is active, filter the ranked jokes by category
        filtered_results = []
        for joke_text, score in ranked_jokes:
            if joke_text in joke_data_map:
                joke_data = joke_data_map[joke_text]
                
                # Apply category filter if specified
                if category and category != 'general' and category != '':
                    if category.lower() not in joke_data.get('category', '').lower():
                        continue  # Skip jokes that don't match the category
                
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
        
        # Format joke texts and scores for response
        jokes_with_scores = []
        for joke in jokes:
            joke_text = ""
            if joke.get('title') and joke.get('body'):
                joke_text = f"{joke['title']}: {joke['body']}"
            elif joke.get('body'):
                joke_text = joke['body']
                
            jokes_with_scores.append({
                "joke": joke_text,
                "score": joke.get('score', 1.0)
            })
            
        return jsonify({
            "jokes_with_scores": jokes_with_scores
        })
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
        "joke_texts": len(joke_texts),
        "categories": jokes_df['category'].dropna().unique().tolist(),
        "sample_jokes": jokes_df.head(3).to_dict('records') if len(jokes_df) > 0 else []
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)