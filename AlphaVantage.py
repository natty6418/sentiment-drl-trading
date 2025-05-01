import requests
import pandas as pd
import time
import os
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import deque

# --- Configuration ---
API_KEY = "Y9VLVP8TVKQBB5LK"
BASE_URL = "https://www.alphavantage.co/query"
TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOG", "META", "BRK-B", "LLY", "V", "TSLA",
    "UNH", "JPM", "XOM", "MA", "AVGO", "PG", "HD", "KO", "MRK", "ORCL",
    "CVX", "ABBV", "PEP", "COST", "ACN", "ADBE", "CRM", "TMO", "WMT", "BAC"
]
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 4, 22)
API_CALL_DELAY_SECONDS = 60.0 / 75.0
OUTPUT_FILE = "dow30_monthly_news_sentiment.csv"

# --- Resume logic ---
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    completed_set = set(zip(existing_df['ticker'], pd.to_datetime(existing_df['published_time']).dt.to_period('M')))
else:
    existing_df = pd.DataFrame()
    completed_set = set()

# --- ETA tracking ---
all_news_data = []
time_records = deque(maxlen=30)  # keep last 30 timings for a rolling average

# --- Date loop ---
date_cursor = START_DATE
total_tasks = len(TICKERS) * ((END_DATE.year - START_DATE.year) * 12 + END_DATE.month - START_DATE.month + 1)
completed_tasks = 0

while date_cursor <= END_DATE:
    next_month = date_cursor + relativedelta(months=1)
    time_from = date_cursor.strftime('%Y%m%dT0000')
    time_to = (next_month - relativedelta(minutes=1)).strftime('%Y%m%dT2359')
    period_label = date_cursor.strftime('%Y-%m')

    print(f"\n--- Collecting articles for {period_label} ---")

    for ticker in TICKERS:
        if (ticker, period_label) in completed_set:
            completed_tasks += 1
            continue

        start_time = time.time()
        print(f"Fetching for {ticker} | Period: {period_label}")
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": time_from,
            "time_to": time_to,
            "sort": "EARLIEST",
            "limit": 1000,
            "apikey": API_KEY
        }

        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if "feed" in data and data["feed"]:
                for article in data["feed"]:
                    ticker_data = next((ts for ts in article.get('ticker_sentiment', []) if ts.get('ticker') == ticker), None)
                    article_data = {
                        'ticker': ticker,
                        'published_time': article.get('time_published'),
                        'title': article.get('title'),
                        'summary': article.get('summary'),
                        'source': article.get('source'),
                        'url': article.get('url'),
                        'overall_sentiment_score': article.get('overall_sentiment_score'),
                        'overall_sentiment_label': article.get('overall_sentiment_label'),
                        'ticker_relevance_score': ticker_data.get('relevance_score') if ticker_data else None,
                        'ticker_sentiment_score': ticker_data.get('ticker_sentiment_score') if ticker_data else None,
                        'ticker_sentiment_label': ticker_data.get('ticker_sentiment_label') if ticker_data else None,
                    }
                    all_news_data.append(article_data)

                # Save data after each ticker-month
                new_df = pd.DataFrame(all_news_data)
                existing_df = pd.concat([existing_df, new_df], ignore_index=True)
                existing_df.to_csv(OUTPUT_FILE, index=False)
                all_news_data.clear()

        except Exception as e:
            print(f"Error for {ticker} in {period_label}: {e}")

        # --- ETA calculation ---
        duration = time.time() - start_time
        time_records.append(duration)
        completed_tasks += 1
        avg_time = sum(time_records) / len(time_records)
        remaining_tasks = total_tasks - completed_tasks
        eta_seconds = int(avg_time * remaining_tasks)
        eta_formatted = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
        print(f"Completed {completed_tasks}/{total_tasks} | ETA: {eta_formatted}")

        time.sleep(API_CALL_DELAY_SECONDS)

    date_cursor = next_month

print("All data collection completed.")
