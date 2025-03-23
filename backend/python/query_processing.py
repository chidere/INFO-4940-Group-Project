import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class QueryProcessor:
  """
  Handles processing of user queries for the joke recommender system.
  Extracts keywords, identifies humor categories, has "Did you mean?" feature, and prepares queries
  for joke retrieval.
  """
  def __init__(self):
    """Initialize the query processor with necessary tools."""
    self.humor_categories = {
      'pun': ['pun', 'wordplay', 'play on words'],
      'one-liner': ['one liner', 'one-liner', 'short', 'quick'],
      'blonde': ['blonde', 'dumb', 'silly'],
      'general': ['general', 'normal', 'regular']
    }
    self.joke_subjects = [
      'food', 'work', 'school', 'politics', 'animals', 'sports',
      'relationships', 'technology', 'health', 'money', 'travel'
    ]
    # TF-IDF vectorizer for keyword extraction
    self.tfidf_vectorizer = TfidfVectorizer(
      stop_words='english',
      max_features=1000,
      ngram_range=(1, 2)
    )
  
  def preprocess_query(self, query):
    """
    Normalize user query.
    
    Args:
      query (str): User input query
        
    Returns:
      str: Preprocessed query
    """
    query = query.lower()
    query = re.sub(r'[^a-zA-Z\s]', '', query)
    tokens = query.split()
    return ' '.join(tokens)
  
  def extract_keywords(self, query):
    """
    Extract important keywords from the query for joke matching.
    
    Args:
      query (str): Preprocessed user query
        
    Returns:
      list: List of important keywords
    """
    words = query.split()
    keywords = [word for word in words if word in self.joke_subjects]
    if not keywords:
      # returns longest words as most significant if no subject keywords found
      words.sort(key=len, reverse=True)
      keywords = words[:3] if len(words) >= 3 else words
    return keywords
  
  def identify_humor_category(self, query):
    """
    Identify the humor category requested in the query.
    
    Args:
      query (str): User query
        
    Returns:
      str: Humor category or 'general' if none found
    """
    query_lower = query.lower()
    for category, keywords in self.humor_categories.items():
      for keyword in keywords:
        if keyword in query_lower:
          return category
    return 'general' # if no category found

  def detect_sentiment(self, query):
    """
    Detect the sentiment or tone requested in the query.
    
    Args:
      query (str): User query
        
    Returns:
      str: Sentiment ('sarcastic', 'clever', 'dark', 'clean', 'neutral')
    """
    query_lower = query.lower()
    sentiment_keywords = {
      'sarcastic': ['sarcastic', 'sarcasm', 'ironic', 'irony'],
      'clever': ['clever', 'witty', 'smart', 'intelligent'],
      'dark': ['dark', 'edgy', 'morbid', 'black humor'],
      'clean': ['clean', 'family friendly', 'pg', 'appropriate']
    }
    for sentiment, keywords in sentiment_keywords.items():
      for keyword in keywords:
        if keyword in query_lower:
          return sentiment
    return 'neutral' #return neutral if no sentiment found
  
  def get_did_you_mean_suggestions(self, query, joke_corpus, max_suggestions=3):
    """
    Generates "Did you mean?" suggestions for potentially mistaken queries.
    
    Args:
      query (str): User query
      joke_corpus (list): List of joke texts to generate suggestions from
      max_suggestions (int): Maximum number of suggestions to return
        
    Returns:
      list: List of query suggestions
    """
    processed_query = self.preprocess_query(query)
    query_keywords = self.extract_keywords(processed_query)
    suggestions = []
    for potential_topic in self.joke_subjects:
      for keyword in query_keywords:
        if (keyword in potential_topic or potential_topic in keyword) and potential_topic not in suggestions:
          suggestions.append(f"jokes about {potential_topic}")
          break
    return suggestions[:max_suggestions]
  
  def process_query(self, query):
    """
    Main method to process a user query and extract all necessary information.
    
    Args:
      query (str): User's original query
        
    Returns:
      dict: Dictionary containing processed query information
    """
    processed_query = self.preprocess_query(query)
    keywords = self.extract_keywords(processed_query)
    category = self.identify_humor_category(query)
    sentiment = self.detect_sentiment(query)
    all_category_keywords = [kw for sublist in self.humor_categories.values() for kw in sublist]
    filtered_keywords = [kw for kw in keywords if kw not in all_category_keywords]
    query_info = {
      'original_query': query,
      'processed_query': processed_query,
      'keywords': filtered_keywords if filtered_keywords else keywords,
      'category': category,
      'sentiment': sentiment
    }
    return query_info