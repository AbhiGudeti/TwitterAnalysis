import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch
import re

class TwitterAnalyzer:
    #Initialize Model
    def __init__(self, bert_model = "nlptown/bert-base-multilingual-uncased-sentiment"):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model)
        self.vader = SentimentIntensityAnalyzer()

    #Scrape tweets within our set time range with specified query parameters
    def scrape_tweets(self, query, days_back = 7, limit = 1000):
        end_date = datetime.now()
        start_date = end_date - timedelta(days = days_back)

        tweets = []
        for tweet in sntwitter.TwitterSearchScraper(
            f'{query} since:{start_date.strftime("%Y-%m-%d")} until:{end_date.strftime("%Y-%m-%d")}'
        ).get_items():
            if len(tweets) >= limit:
                break
            tweets.append({
                'date': tweet.date,
                'text': tweet.rawContent,
                'username': tweet.user.username,
                'likes': tweet.likeCount,
                'retweets': tweet.retweetCount
            })
        
        return pd.DataFrame(tweets)
    
    #Preprocess our string for sentiment analysis
    def preprocess_text(self, text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags = re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)

        return text.lower().strip()

    #Get a BERT model sentiment score
    def get_bert_sentiment(self, text):
        inputs = self.bert_tokenizer(text, return_tensors = "pt", truncation = True, max_length = 512)
        outputs = self.bert_model(**inputs)

        return int(torch.argmax(outputs.logits)) + 1
    
    #Get a VADER sentiment score
    def get_vader_sentiment(self, text):
        return self.vader.polarity_scores(text)

    #TO IMPLEMENT HYBRID SENTIMENT ANALYSIS
