import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import glob
import subprocess
import time

# Function to start subprocesses
def start_subprocesses():
    st.write("Starting data collection process...")
    subprocess.Popen(["python3", "data_collection.py"])
    time.sleep(5)

    st.write("Starting data consumer process...")
    subprocess.Popen(["spark-submit", "data_consumer.py"])
    time.sleep(5)

    st.write("Starting reference data processing...")
    subprocess.Popen(["spark-submit", "data_processing_reference.py"])
    time.sleep(1)

    st.write("Starting sentiment data processing...")
    subprocess.Popen(["python3", "data_processing_sentiment.py"])
    time.sleep(1)

    st.write("Starting TF-IDF data processing...")
    subprocess.Popen(["spark-submit", "data_processing_tfidf.py"])

# Function to read Parquet files and concatenate them into a DataFrame
def load_data_from_parquet(path):
    files = glob.glob(path + "/*.parquet")
    df_list = [pd.read_parquet(file) for file in files]
    return pd.concat(df_list, ignore_index=True)

# Start subprocesses when the button is clicked
if st.button('Start Data Collection and Processing'):
    start_subprocesses()
    st.success('Subprocesses started successfully!')

# Load and visualize data
st.title('Sentiment Analysis Dashboard')

try:
    # Load the raw data
    raw_data_path = "data/raw"
    raw_data = load_data_from_parquet(raw_data_path)

    # Load the sentiment data
    sentiment_data_path = "data/sentiment"
    sentiment_data = load_data_from_parquet(sentiment_data_path)

    # Load the reference data
    reference_data_path = "data/reference"
    reference_data = load_data_from_parquet(reference_data_path)

    # Load the TF-IDF data
    tfidf_data_path = "data/tfidf"
    tfidf_data = load_data_from_parquet(tfidf_data_path)

    # Merge and clean data
    all_data = pd.merge(sentiment_data, raw_data, how='left', on='id')
    all_data.dropna(inplace=True)

    # Drop duplicated columns if you only want to keep one set
    columns_to_drop = ['author_x', 'created_utc_x', 'score_x', 'parent_id_x',
                       'subreddit_x', 'permalink_x', 'text_x', 'timestamp_x']
    all_data = all_data.drop(columns=columns_to_drop)
    all_data = all_data.rename(columns={
        'author_y': 'author',
        'created_utc_y': 'created_utc',
        'score_y': 'score',
        'parent_id_y': 'parent_id',
        'subreddit_y': 'subreddit',
        'permalink_y': 'permalink',
        'text_y': 'text',
        'timestamp_y': 'timestamp'
    })

    # Word Cloud
    st.header('Word Cloud')
    all_words = ' '.join([' '.join(text) for text in all_data['finished_no_stop_lemmatized']])
    word_cloud = WordCloud(width=800, height=400, background_color='black').generate(all_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Histogram of compound values
    st.header('Histogram of Compound Scores')
    plt.figure(figsize=(10, 5))
    sns.histplot(all_data['compound'], bins=30, kde=True)
    plt.title('Histogram of Compound Scores')
    plt.xlabel('Compound Score')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    # Bar chart of sentiment scores
    st.header('Average Sentiment Scores')
    sentiment_means = all_data[['neutral', 'positive', 'negative']].mean()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=sentiment_means.index, y=sentiment_means.values)
    plt.title('Average Sentiment Scores')
    plt.xlabel('Sentiment')
    plt.ylabel('Average Score')
    st.pyplot(plt)

    # Time Series Visualization using groupby
    st.header('Compound Score Over Time')
    all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])
    all_data.sort_values('timestamp', inplace=True)

    # Extract the hour from the timestamp for grouping
    all_data['hour'] = all_data['timestamp'].dt.floor('H')

    # Group by the hour and calculate the mean compound score
    grouped_data = all_data.groupby('hour')['compound'].mean().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(grouped_data['hour'], grouped_data['compound'], marker='o', linestyle='-', markersize=2, label='Compound Score')
    plt.title('Compound Score Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Compound Score')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

    # Scatter plot of compound score vs score
    st.header('Scatter Plot of Compound Score vs Score')
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='score', y='compound', data=all_data)
    plt.title('Scatter Plot of Compound Score vs Score')
    plt.xlabel('Score')
    plt.ylabel('Compound Score')
    st.pyplot(plt)

    # Display reference data
    st.header('Reference Data')
    st.write(reference_data)

    # Display TF-IDF data
    st.header('TF-IDF Data')
    st.write(tfidf_data)

except Exception as e:
    st.error(f"Error loading or processing data: {e}")
