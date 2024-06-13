import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import socket
import json
import time
import threading
import os

# Initialize global DataFrames
all_data = pd.DataFrame()
sentiment_data = pd.DataFrame()

# Function to update data continuously
def update_data():
    global all_data, sentiment_data

    # Connect to the socket
    host = "localhost"
    port = 9998
    s = socket.socket()
    s.connect((host, port))

    while True:
        try:
            # Receive data from the socket
            data = s.recv(1024)
            if not data:
                break

            # Parse the JSON data
            json_data = json.loads(data.decode('utf-8'))

            # Append the new data to the DataFrame
            new_data = pd.DataFrame([json_data])
            all_data = pd.concat([all_data, new_data], ignore_index=True)



            # Process sentiment data
            new_sentiment = {
                'id': json_data['id'],
                'compound': json_data['score'] / 100,  # Simulated sentiment score
                'neutral': json_data['score'] / 200,   # Simulated sentiment score
                'positive': json_data['score'] / 300,  # Simulated sentiment score
                'negative': json_data['score'] / 400,  # Simulated sentiment score
                'finished_no_stop_lemmatized': json_data['text'].split(),
                'timestamp': pd.to_datetime(json_data['created_utc'], unit='s')
            }
            sentiment_data = pd.concat([sentiment_data, pd.DataFrame([new_sentiment])], ignore_index=True)

            # Limit the size of the DataFrame to the last 1000 rows
            if len(all_data) > 1000:
                all_data = all_data.tail(1000)
            if len(sentiment_data) > 1000:
                sentiment_data = sentiment_data.tail(1000)

        except Exception as e:
            print(f"Error receiving data: {e}")
            break

# Start the data update thread
data_thread = threading.Thread(target=update_data)
data_thread.daemon = True
data_thread.start()

# Title of the app
st.title('Real-Time Sentiment Analysis Dashboard')

# Placeholders for charts
bar_chart_placeholder = st.empty()
time_series_placeholder = st.empty()
word_cloud_placeholder = st.empty()
histogram_placeholder = st.empty()
scatter_plot_placeholder = st.empty()
reference_data_placeholder = st.empty()
tfidf_data_placeholder = st.empty()

# Function to read Parquet files and concatenate them into a DataFrame
def load_data_from_parquet(directory):
    dataframes = []
    for file in os.listdir(directory):
        if file.endswith('.parquet'):
            file_path = os.path.join(directory, file)
            df = pd.read_parquet(file_path)
            dataframes.append(df)
    return pd.concat(dataframes)

while True:
    if not sentiment_data.empty:
        try:
            # Bar chart of sentiment scores
            sentiment_means = sentiment_data[['neutral', 'positive', 'negative']].mean()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=sentiment_means.index, y=sentiment_means.values, ax=ax)
            ax.set_title('Average Sentiment Scores')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Average Score')
            bar_chart_placeholder.pyplot(fig)

            # Time Series Visualization
            grouped_data = sentiment_data.groupby('timestamp')['compound'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(grouped_data['timestamp'], grouped_data['compound'], marker='o', linestyle='-', markersize=2, label='Compound Score')
            ax.set_title('Compound Score Over Time')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Compound Score')
            ax.legend()
            time_series_placeholder.pyplot(fig)

            # Word Cloud
            all_words = ' '.join([' '.join(text) for text in sentiment_data['finished_no_stop_lemmatized']])
            word_cloud = WordCloud(width=800, height=400, background_color='black').generate(all_words)
            plt.figure(figsize=(10, 5))
            plt.imshow(word_cloud, interpolation='bilinear')
            plt.axis('off')
            word_cloud_placeholder.pyplot(plt)

            # Histogram of compound values
            plt.figure(figsize=(10, 5))
            sns.histplot(sentiment_data['compound'], bins=30, kde=True)
            plt.title('Histogram of Compound Scores')
            plt.xlabel('Compound Score')
            plt.ylabel('Frequency')
            histogram_placeholder.pyplot(plt)

            # Scatter plot of compound score vs score
            #plt.figure(figsize=(10, 5))
            #sns.scatterplot(x='score', y='compound', data=sentiment_data)
            #plt.title('Scatter Plot of Compound Score vs Score')
            #plt.xlabel('Score')
            #plt.ylabel('Compound Score')
            #scatter_plot_placeholder.pyplot(plt)

            # Load and display reference data
            reference_data_path = "data/reference"
            reference_data = load_data_from_parquet(reference_data_path)
            reference_data_placeholder.write(reference_data)

            # Load and display TF-IDF data
            tfidf_data_path = "data/tfidf"
            tfidf_data = load_data_from_parquet(tfidf_data_path)
            tfidf_data_placeholder.write(tfidf_data)

        except Exception as e:
            st.error(f"Error updating charts: {e}")

    # Add a short delay to avoid overwhelming the CPU
    time.sleep(1)

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

# Start subprocesses when the button is clicked
if st.button('Start Data Collection and Processing'):
    start_subprocesses()
    st.success('Subprocesses started successfully!')
