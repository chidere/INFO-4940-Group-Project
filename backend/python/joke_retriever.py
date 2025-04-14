from python.query_processing import QueryProcessor
from profanity_check import predict

def is_inappropriate(text):
    """Checks if the joke has profanity in it using the python profanity library (returns 1 for inappropriate, 0 for appropriate)"""
    result = predict([text])
    return result[0] == 1 

def retrieve_jokes(category, dataset):
    """Retrieves jokes based on the category while filtering out inappropriate content."""
    relevant_jokes = [
        joke for joke in dataset 
        if category.lower() in joke['category'].lower() and not  is_inappropriate(joke['joke_text'])
    ]
    return relevant_jokes

#def retrieve_jokes(category, dataset, inappropriate_words):
    """Retrieves jokes based on the category while filtering out inappropriate content."""
    relevant_jokes = [
        joke for joke in dataset 
        if category.lower() in joke['category'].lower() and not any(word in joke['joke_text'].lower() for word in inappropriate_words)
    ]
    return relevant_jokes


def main():
    query_processor = QueryProcessor()
    
    # Example dataset
    dataset = [
        {"id": 1, "joke_text": "Why don’t skeletons fight each other? They don’t have the guts.", "category": "Puns", "rating": 5},
        {"id": 2, "joke_text": "I told my wife she was drawing her eyebrows too high. She looked surprised.", "category": "One-Liners", "rating": 4},
        {"id": 3, "joke_text": "Why did the scarecrow win an award? Because he was outstanding in his field!", "category": "Puns", "rating": 4},
        {"id": 4, "joke_text": "This joke contains an inappropriate word.", "category": "General", "rating": 3}
    ]
    
    #inappropriate_words = {"fuck", "shit", "bitch", "cunt", "pussy", "shit"}  # Example filter words
    
    # User input for query
    user_query = input("Enter your joke request: ")
    processed_query = query_processor.process_query(user_query)
    category = processed_query['category']
    
    #jokes = retrieve_jokes(category, dataset, inappropriate_words)

    jokes = retrieve_jokes(category, dataset)

    if jokes:
        print("Jokes found:")
        for joke in jokes:
            print(f"- {joke['joke_text']} (Rating: {joke['rating']})")
    else:
        print("No appropriate jokes found for this category.")

if __name__ == "__main__":
    main()