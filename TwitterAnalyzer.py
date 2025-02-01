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
    def hybrid_sentiment_analysis(self, df):
        df['processed_text'] = df['text'].apply(self.preprocess_text)

        df['vader_scores'] = df['processed_text'].apply(self.get_vader_sentiment)
        df['vader_compound'] = df['vader_scores'].apply(lambda x: x['compound'])

        df['bert_sentiment'] = df['processed_text'].apply(self.get_bert_sentiment)

        df['final_sentiment'] = (0.6 * ((df['bert_sentiment'] - 1) / 4) + 0.4 * ((df['vader_compound'] + 1) / 2))

        return df


    #Latent Dirichlet Allocation text extraction
    def extract_topics(self, df, n_topics = 10):
        vectorizer = CountVectorizer(max_df = 0.95, min_df = 2, stop_words = 'english')
        doc_term_matrix = vectorizer.fit_transform(df['processed_text'])

        lda = LatentDirichletAllocation(
            n_components = n_topics,
            random_state = 42,
            learning_method = 'online'
        )

        lda_output = lda.fit_transform(doc_term_matrix)
        feature_names = vectorizer.get_feature_names_out()

        topics = []
        for id, topic in enumerate(lda.components_):
            top_words = [feature_names[x] for x in topic.argsort()[:-10-1:-1]]
            topics.append({
                'topic': f'Topic {id + 1}',
                'words': top_words
            })
        
        return pd.DataFrame(topics)
    
    #Visualization
    def create_vis(self, df):
        #Sentiment Distribution
        fig_sentiment = px.histogram(
            df,
            x = 'final_sentiment',
            title = 'Sentiment Distribution',
            labels = {'final_sentiment', 'Sentiment Score'},
            template = 'seaborn'
        )

        #Sentiment Over Time
        daily_sentiment = df.groupby(df['date'].dt.date)['final_sentiment'].mean().reset_index()
        fig_timeline = px.line(
            daily_sentiment,
            x = 'date',
            y = 'final_sentiment',
            title = 'Sentiment Trend Over Time',
            template = 'seaborn'
        )

        #Engagement Correlation
        fig_engagement = px.scatter(
            df,
            x = 'final_sentiment',
            y = 'likes',
            size = 'retweets',
            title = 'Sentiment vs Engagement',
            template = 'seaborn'
        )

        return {
            'distr': fig_sentiment,
            'timeline': fig_timeline,
            'engagement': fig_engagement
        }
    
    def main():
        analyzer = TwitterAnalyzer()
        query = "NVIDIA" #Entrer brand name here

        tweets_df = analyzer.scrape_tweets(query)

        analyzed_df = analyzer.hybrid_sentiment_analysis(tweets_df)
        topics_df = analyzer.extract_topics(analyzed_df)
        visualizations = analyzer.create_vis(analyzed_df)

        analyzed_df.to_csv('sentiment_analysis.csv', index = False)
        topics_df.to_csv('topics.csv', index = False)

        for name, fig in visualizations.items():
            fig.write_html(f'{name}.html')
    

    if __name__ == "__main__":
        main()
