import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from python.query_processing import QueryProcessor
from python.joke_ranker import JokeRanker
from python.joke_retriever import retrieve_jokes

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    reddit_df = pd.DataFrame(data['reddit_jokes'])
    stupidstuff_df = pd.DataFrame(data['stupidstuff'])
    wocka_df = pd.DataFrame(data['wocka'])

app = Flask(__name__)
CORS(app)

query_processor = QueryProcessor()

temp_jokes_path = os.path.join(current_directory, 'temp_jokes.csv')
combined_jokes = []

for df in [reddit_df, stupidstuff_df, wocka_df]:
    for _, row in df.iterrows():
        joke_text = f"{row.get('title', '')} {row.get('body', '')}"
        combined_jokes.append({"joke": joke_text.strip()})

pd.DataFrame(combined_jokes).to_csv(temp_jokes_path, index=False)

joke_ranker = JokeRanker(temp_jokes_path)

def json_search(query):
    matches = []
    
    for df in [reddit_df, stupidstuff_df, wocka_df]:
        df_matches = df[df['title'].str.lower().str.contains(query.lower(), na=False)]
        matches.extend(df_matches.to_dict('records'))
        df_matches = df[df['body'].str.lower().str.contains(query.lower(), na=False)]
        matches.extend(df_matches.to_dict('records'))
    
    unique_ids = set()
    unique_matches = []
    for match in matches:
        if match['id'] not in unique_ids:
            unique_ids.add(match['id'])
            unique_matches.append(match)
    
    return json.dumps(unique_matches)

def joke_search(query):
    """
    Search for jokes based on a query string.
    Uses QueryProcessor, JokeRanker, and retrieve_jokes.
    """
    query_info = query_processor.process_query(query)
    category = query_info['category']
    
    all_jokes = []
    for df in [reddit_df, stupidstuff_df, wocka_df]:
        all_jokes.extend(df.to_dict('records'))
    
    category_jokes = retrieve_jokes(category, all_jokes)
    
    # If not enough jokes by category, use the ranker
    if len(category_jokes) < 5:
        search_query = ' '.join(query_info['keywords'])
        ranked_jokes = joke_ranker.rank_jokes(search_query, 5)
        
        results = []
        for joke_text, score in ranked_jokes:
            for df in [reddit_df, stupidstuff_df, wocka_df]:
                matches = df[df.apply(
                    lambda row: joke_text.startswith(f"{row.get('title', '')} {row.get('body', '')}"),
                    axis=1
                )]
                
                if not matches.empty:
                    joke_data = matches.iloc[0].to_dict()
                    joke_data['score'] = float(score)
                    results.append(joke_data)
                    break
        
        return results 
    
    else:
        return category_jokes[:5]

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

# @app.route("/roast-it")
# def search_jokes():
#     query = request.args.get("query", "")
#     if not query:
#         return jsonify({"error": "No query provided"}), 400
    
#     try:
#         query_info = query_processor.process_query(query)

#         joke_texts = []
        
#         for _, row in stupidstuff_df.head(5).iterrows():
#             if 'title' in row and 'body' in row:
#                 joke_texts.append(f"{row['title']}: {row['body']}")
        
#         if len(joke_texts) < 5:
#             remaining = 5 - len(joke_texts)
#             for _, row in stupidstuff_df.head(remaining).iterrows():
#                 if 'title' in row and 'body' in row:
#                     joke_texts.append(f"{row['title']}: {row['body']}")
        
#         return jsonify({
#             "jokes": joke_texts
#         })
        
#     except Exception as e:
#         print(f"Error in search_jokes: {str(e)}")
#         return jsonify({
#             "error": str(e),
#             "jokes": []
#         }), 500

@app.route("/roast-it")
def search_jokes():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # To test if this works
        sample_jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!"
        ]
        
        return jsonify({
            "jokes": sample_jokes
        })
    except Exception as e:
        print(f"Error in search_jokes: {str(e)}")
        return jsonify({
            "error": str(e),
            "jokes": []
        }), 500
    
if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)