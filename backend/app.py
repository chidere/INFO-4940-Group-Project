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

def joke_search(query):
    try:
        # Process the query to extract information
        query_info = query_processor.process_query(query)
        print(f"Processed query: {query_info}")
        
        # Get category from query
        category = query_info['category']
        
        # Try to find jokes in category
        category_jokes = []
        if category != 'general':
            category_jokes = jokes_df[jokes_df['category'].str.contains(
                category, case=False, na=False)].to_dict('records')
        
        # If not enough category jokes, use ranking
        if len(category_jokes) < 5:
            search_query = ' '.join(query_info['keywords'])
            print(f"Searching with keywords: {search_query}")
            
            # Get ranked jokes
            ranked_jokes = joke_ranker.rank_jokes(search_query, 5)
            print(f"Found {len(ranked_jokes)} ranked jokes")
            
            results = []
            for joke_text, score in ranked_jokes:
                if joke_text in joke_data_map:
                    joke_data = joke_data_map[joke_text]
                    results.append({
                        'title': joke_data.get('title', ''),
                        'body': joke_data.get('body', ''),
                        'category': joke_data.get('category', ''),
                        'score': float(score)
                    })
            return results
        else:
            # Return category jokes
            for joke in category_jokes:
                joke['score'] = 1.0
            return category_jokes[:5]
    except Exception as e:
        print(f"Error in joke_search: {str(e)}")
        return []

@app.route("/")
def home():
    return render_template('base.html', title="Joke Recommender")

@app.route("/roast-it")
def search_jokes():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        jokes = joke_search(query)
        
        # Format joke texts for response
        joke_texts = []
        for joke in jokes:
            if joke.get('title') and joke.get('body'):
                joke_texts.append(f"{joke['title']}: {joke['body']}")
            elif joke.get('body'):
                joke_texts.append(joke['body'])
        return jsonify({
            "jokes": joke_texts
        })
    except Exception as e:
        print(f"Error in search_jokes: {str(e)}")
        return jsonify({
            "error": str(e),
            "jokes": []
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
