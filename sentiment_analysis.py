import pandas as pd
import re
import string
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\polas\OneDrive\Desktop\INTERNSHIP(Next24tech)\TASK 3\data/tweets.csv")

# Clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+|#", "", text)   # Remove @mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text


df['cleaned_text'] = df['text'].apply(clean_text)

# Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

# Visualize results
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Analysis of Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.savefig('sentiment_distribution.png')
plt.show()

# Save output
df.to_csv('data/tweets_with_sentiment.csv', index=False)
