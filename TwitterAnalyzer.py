import tweepy
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
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TwitterAnalyzer:
    def __init__(self, bert_model="nlptown/bert-base-multilingual-uncased-sentiment"):
        # Twitter API authentication
        self.client = tweepy.Client(
            bearer_token=os.environ.get('TWITTER_BEARER_TOKEN'),
            wait_on_rate_limit=True
        )
        
        # Initialize models
        logger.info("Loading BERT model...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model)
        self.vader = SentimentIntensityAnalyzer()
        
    def scrape_tweets(self, query, days_back=7, limit=1000):
        try:
            logger.info(f"Fetching tweets for query: {query}")
            end_time = datetime.now() - timedelta(minutes = 1)
            start_time = end_time - timedelta(days = days_back)

            tweets = []
            for response in tweepy.Paginator(
                self.client.search_recent_tweets,
                query = query,
                max_results = 100,
                start_time = start_time,
                end_time = end_time,
                tweet_fields = ['created_at', 'public_metrics']
            ):
                if response.data:
                    for tweet in response.data:
                        tweets.append({
                            'date': tweet.created_at,
                            'text': tweet.text,
                            'likes': tweet.public_metrics['like_count'],
                            'retweets': tweet.public_metrics['retweet_count']
                        })
                
                if len(tweets) >= limit:
                    break
            
            return pd.DataFrame(tweets)

        except Exception as e:
            logger.error(f"Error fetching tweets: {str(e)}")
            raise

    #Clean tweet text
    def preprocess_text(self, text):
        try:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'\@\w+|\#\w+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            return text.lower().strip()
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            raise

    #Get BERT sentiment score
    @torch.no_grad()  # Disable gradient calculation for inference
    def get_bert_sentiment(self, text):
        try:
            inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.bert_model(**inputs)
            return int(torch.argmax(outputs.logits)) + 1
        except Exception as e:
            logger.error(f"Error in BERT sentiment analysis: {str(e)}")
            raise

    #Get VADER Sentiment Scores
    def get_vader_sentiment(self, text):
        try:
            return self.vader.polarity_scores(text)
        except Exception as e:
            logger.error(f"Error in VADER sentiment analysis: {str(e)}")
            raise

    #Combine BERT and VADER sentiment analysis
    def hybrid_sentiment_analysis(self, df):
        try:
            logger.info("Starting hybrid sentiment analysis...")
            
            if df.empty:
                raise ValueError("Empty DataFrame provided")
                
            df['processed_text'] = df['text'].apply(self.preprocess_text)
            
            # VADER analysis
            df['vader_scores'] = df['processed_text'].apply(self.get_vader_sentiment)
            df['vader_compound'] = df['vader_scores'].apply(lambda x: x['compound'])
            
            # BERT analysis
            df['bert_sentiment'] = df['processed_text'].apply(self.get_bert_sentiment)
            
            # Combine scores (weighted average)
            df['final_sentiment'] = (
                0.6 * ((df['bert_sentiment'] - 1) / 4) + 
                0.4 * ((df['vader_compound'] + 1) / 2)
            )
            
            logger.info("Completed sentiment analysis")
            return df
            
        except Exception as e:
            logger.error(f"Error in hybrid sentiment analysis: {str(e)}")
            raise

    #Extract topics using LDA
    def extract_topics(self, df, n_topics=10):
        try:
            logger.info(f"Extracting {n_topics} topics...")
            
            vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            doc_term_matrix = vectorizer.fit_transform(df['processed_text'])
            
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                learning_method='online'
            )
            
            lda_output = lda.fit_transform(doc_term_matrix)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top words per topic
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-10-1:-1]]
                topics.append({
                    'topic': f'Topic {topic_idx + 1}',
                    'words': top_words
                })
            
            logger.info("Completed topic extraction")
            return pd.DataFrame(topics)
            
        except Exception as e:
            logger.error(f"Error in topic extraction: {str(e)}")
            raise

    #Generate interactive visualizations
    def create_visualizations(self, df):
        try:
            logger.info("Creating visualizations...")
            
            # Sentiment distribution
            fig_sentiment = px.histogram(
                df, 
                x='final_sentiment',
                title='Sentiment Distribution',
                labels={'final_sentiment': 'Sentiment Score'},
                template='plotly_white'
            )
            
            # Sentiment over time
            daily_sentiment = df.groupby(df['date'].dt.date)['final_sentiment'].mean().reset_index()
            fig_timeline = px.line(
                daily_sentiment,
                x='date',
                y='final_sentiment',
                title='Sentiment Trend Over Time',
                template='plotly_white'
            )
            
            # Engagement correlation
            fig_engagement = px.scatter(
                df,
                x='final_sentiment',
                y='likes',
                size='retweets',
                title='Sentiment vs. Engagement',
                template='plotly_white'
            )
            
            logger.info("Completed creating visualizations")
            return {
                'sentiment_dist': fig_sentiment,
                'sentiment_timeline': fig_timeline,
                'engagement': fig_engagement
            }
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise

def main():
    try:
        # Initialize analyzer
        analyzer = TwitterAnalyzer()
        
        # Example usage
        query = "your_brand_name"
        tweets_df = analyzer.scrape_tweets(query)
        
        if tweets_df.empty:
            logger.warning("No tweets found for the given query")
            return
        
        # Run analysis
        analyzed_df = analyzer.hybrid_sentiment_analysis(tweets_df)
        topics_df = analyzer.extract_topics(analyzed_df)
        visualizations = analyzer.create_visualizations(analyzed_df)
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Save results
        analyzed_df.to_csv('output/sentiment_analysis.csv', index=False)
        topics_df.to_csv('output/topics.csv', index=False)
        
        # Save visualizations
        for name, fig in visualizations.items():
            fig.write_html(f'output/{name}.html')
            
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()