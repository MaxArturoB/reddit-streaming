import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

raw_path = r"C:\Users\fagos\PycharmProjects\pythonProject\RealTime\data\raw"
sentiment_path = r"C:\Users\fagos\PycharmProjects\pythonProject\RealTime\data\sentiment"
tf_idf_path = r"C:\Users\fagos\PycharmProjects\pythonProject\RealTime\data\tfidf"


def read_parquet(directory):
    dataframes = []
    for file in os.listdir(directory):
        if file.endswith('.parquet'):
            file_path = os.path.join(directory, file)
            df = pd.read_parquet(file_path)
            dataframes.append(df)
    return pd.concat(dataframes)


raw_data = read_parquet(raw_path)
sentiment_data = read_parquet(sentiment_path)
tfidf_data = read_parquet(tf_idf_path)
# Load your data


# Assuming sentiment_data and raw_data are loaded here

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

# Title of the app
st.title('Sentiment Analysis Dashboard')

# Word Cloud
st.header('Word Cloud')
all_words = ' '.join([' '.join(text) for text in all_data['finished_no_stop_lemmatized']])
word_cloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
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
