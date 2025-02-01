# Twitter Campaign Sentiment Analyzer

A Python tool for analyzing brand sentiment and campaign performance on Twitter using advanced NLP techniques.

## Features

- Hybrid sentiment analysis combining BERT and VADER
- Topic extraction using Latent Dirichlet Allocation (LDA)
- Interactive visualizations with Plotly
- Customizable tweet scraping
- Engagement metrics analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from twitter_analyzer import TwitterAnalyzer

# Initialize analyzer
analyzer = TwitterAnalyzer()

# Analyze tweets
tweets_df = analyzer.scrape_tweets("your_brand_name")
analyzed_df = analyzer.hybrid_sentiment_analysis(tweets_df)
topics_df = analyzer.extract_topics(analyzed_df)
visualizations = analyzer.create_visualizations(analyzed_df)
```

## Key Components

- **TwitterAnalyzer**: Main class handling all analysis functionality
- **scrape_tweets()**: Collects tweets based on query and time range
- **hybrid_sentiment_analysis()**: Combines BERT and VADER for enhanced accuracy
- **extract_topics()**: Identifies key topics using LDA
- **create_visualizations()**: Generates interactive Plotly visualizations

## Output

- `sentiment_analysis.csv`: Detailed sentiment analysis results
- `topics.csv`: Extracted topics and associated keywords
- Interactive HTML visualizations:
  - Sentiment distribution
  - Sentiment trends over time
  - Engagement correlation charts

## Requirements

- Python 3.8+
- transformers
- torch
- vaderSentiment
- scikit-learn
- plotly
- pandas
- snscrape

