# api.py
from fastapi import FastAPI, HTTPException
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import random
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from fastapi.middleware.cors import CORSMiddleware # Import CORS
# api.py (Top section fix)
from fastapi import FastAPI
from contextlib import asynccontextmanager
import nltk

nltk.data.path.append("/nltk_data")  # Explicit NLTK path

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialization code here if needed
    yield  # Keeps context alive

app = FastAPI(lifespan=lifespan)  # Explicit lifespan handler


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - VERY permissive, configure for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


sia = SentimentIntensityAnalyzer()

# Function to extract news
def extract_news(company_name: str) -> List[Dict]:
    rss_url = f"https://news.google.com/rss/search?q={company_name}"
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
    ]

    headers = {"User-Agent": random.choice(user_agents)}

    try:
        response = requests.get(rss_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        articles = []
        articles = []
        for item in soup.find_all("item")[:10]:
            title = item.find("title").text if item.find("title") else "No title found"
            
            summary_content = item.find("description").text if item.find("description") else "Unknown summary"
            summary_soup = BeautifulSoup(summary_content, "html.parser")
            
            link_element = summary_soup.find("a")
            link = link_element["href"] if link_element and "href" in link_element.attrs else "No link found"
            
            summary = summary_soup.get_text()
            
            articles.append({
                "title": title,
                "link": link,
                "summary": summary
            })
        
        return articles

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Function for sentiment analysis
def sentiment_analysis(text: str) -> str:
    sentiment = sia.polarity_scores(text)

    if sentiment['compound'] > 0.05:
        return "Positive"
    elif sentiment['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Function for comparative analysis
def comparative_analysis(articles: List[Dict]) -> Dict:
    sentiments = [sentiment_analysis(article['summary']) for article in articles]

    positive_count = sentiments.count("Positive")
    negative_count = sentiments.count("Negative")
    neutral_count = sentiments.count("Neutral")

    return {
        "Positive": positive_count,
        "Negative": negative_count,
        "Neutral": neutral_count
    }

def extract_topics(summary: str, n: int = 5) -> List[str]:
    """Extract relevant and concise topics using TF-IDF."""

    # Clean the summary
    summary = re.sub(r'[^\w\s]', '', summary).lower()

    # Use TF-IDF to find important words
    vectorizer = TfidfVectorizer(stop_words='english', max_features=n)  # Limit to top n words
    tfidf_matrix = vectorizer.fit_transform([summary])  # Fit and transform

    # Get the top n keywords
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = sorted(zip(tfidf_matrix.toarray()[0], feature_array), reverse=True)[:n]

    # Return the keywords
    keywords = [word for _, word in tfidf_sorting]
    return keywords

# Function to generate comparisons
def generate_comparison(articles: List[Dict]) -> List[Dict]:
    comparisons = []

    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            article1 = articles[i]
            article2 = articles[j]

            comparison = generate_comparison_text(article1, article2, articles)  # Pass all_articles
            impact = generate_impact_text(article1, article2, articles)        # Pass all_articles

            comparisons.append({
                "Comparison": comparison,
                "Impact": impact
            })

    return comparisons

def generate_comparison_text(article1: Dict, article2: Dict, all_articles: List[Dict]) -> str:
    topics1 = set(article1["Topics"])
    topics2 = set(article2["Topics"])
    common_topics = topics1.intersection(topics2)
    unique_topics1 = topics1 - topics2
    unique_topics2 = topics2 - topics1

    # Sentiment Analysis
    if article1["Sentiment"] == article2["Sentiment"]:
        sentiment_part = f"Both articles share a {article1['Sentiment']} sentiment."
    else:
        sentiment_part = f"Article 1 has a {article1['Sentiment']} sentiment, while Article 2 has a {article2['Sentiment']} sentiment."

    # Common Topics
    if common_topics:
        common_topics_part = f"They both discuss {', '.join(common_topics)}."
    else:
        common_topics_part = "They do not share common topics."

    # Unique Topics
    if unique_topics1:
        unique_topics1_part = f"Article 1 uniquely covers {', '.join(unique_topics1)}."
    else:
        unique_topics1_part = "Article 1 does not have unique topics."

    if unique_topics2:
        unique_topics2_part = f"Article 2 uniquely covers {', '.join(unique_topics2)}."
    else:
        unique_topics2_part = "Article 2 does not have unique topics."

    # Broader Trend Consideration
    all_topics = [topic for article in all_articles for topic in article["Topics"]]
    topic_counts = Counter(all_topics)
    trending_topics = [topic for topic, count in topic_counts.most_common(3)]  # Top 3 trending

    if common_topics.intersection(trending_topics):
        trend_part = f"Notably, these articles touch on trending topics such as {', '.join(common_topics.intersection(trending_topics))}."
    else:
        trend_part = "These articles do not focus on the most trending topics."

    comparison = f"{article1['title']} focuses on {', '.join(article1['Topics'])}, while {article2['title']} discusses {', '.join(article2['Topics'])}. {sentiment_part} {common_topics_part} {unique_topics1_part} {unique_topics2_part} {trend_part}"

    return comparison

def generate_impact_text(article1: Dict, article2: Dict, all_articles: List[Dict]) -> str:
    # Sentiment-Based Impact
    positive_count = sum(1 for article in all_articles if article["Sentiment"] == "Positive")
    negative_count = sum(1 for article in all_articles if article["Sentiment"] == "Negative")
    neutral_count = sum(1 for article in all_articles if article["Sentiment"] == "Neutral")
    total_articles = len(all_articles)

    overall_sentiment_ratio = positive_count / total_articles if total_articles > 0 else 0

    if overall_sentiment_ratio > 0.6:
        impact = "Overall positive sentiment suggests potential for investor confidence and market growth."
    elif overall_sentiment_ratio < 0.4:
        impact = "Predominantly negative sentiment may lead to investor caution and potential market downturn."
    else:
        impact = "Mixed sentiment presents a balanced view, likely leading to a period of market stability or uncertainty."

    # Consideration of Conflicting Information
    if article1["Sentiment"] != article2["Sentiment"]:
        impact += " The conflicting sentiments between the articles may create market volatility."

    return impact

@app.get("/extract_news")
async def get_news(company_name: str):
    """Extracts news articles for a given company."""
    try:
        articles = extract_news(company_name)
        return articles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/analyze_articles")
async def analyze_articles(articles: List[Dict]):
    """Analyzes a list of articles, extracting topics, sentiments, and generating comparisons."""

    # 1. Sentiment Analysis and Topic Extraction
    for article in articles:
        article['Sentiment'] = sentiment_analysis(article['summary'])
        article['Topics'] = extract_topics(article['summary'])
        if 'title' in article:
            article['title'] = article.pop('title')
        else:
            article['title'] = "No Title Found"

    # 2. Comparative Sentiment Score
    sentiments = comparative_analysis(articles)

    # 3. Generate Comparisons
    comparisons = generate_comparison(articles)

    # 4. Topic Overlap
    topics_per_article = {i: article['Topics'] for i, article in enumerate(articles)}
    all_topics = set()
    for topics in topics_per_article.values():
        all_topics.update(topics)

    common_topics = []
    for topic in all_topics:
        count = sum(topic in article_topics for article_topics in topics_per_article.values())
        if count > 1:
            common_topics.append((topic, count))

    unique_topics = {}
    for i in range(len(articles)):
        other_topics = set()
        for j in range(len(articles)):
            if i != j:
                other_topics.update(topics_per_article[j])
        unique_topics[f"Article {i+1}"] = list(set(topics_per_article[i]) - other_topics)

    # 5. Final Sentiment Analysis
    if sentiments["Positive"] > sentiments["Negative"]:
        final_sentiment = f"Latest news coverage is mostly positive. Potential stock growth expected."
    else:
        final_sentiment = f"Latest news coverage is mostly negative. Potential stock decline expected."

    output_data = {
        "Articles": articles,
        "Comparative Sentiment Score": {
            "Sentiment Distribution": sentiments,
            "Coverage Differences": comparisons,
            "Topic Overlap": {
                "Common Topics": [topic for topic, count in common_topics],
                "Unique Topics": unique_topics
            }
        },
        "Final Sentiment Analysis": final_sentiment,
    }

    return output_data
