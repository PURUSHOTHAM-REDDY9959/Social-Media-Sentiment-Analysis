# Social-Media-Sentiment-Analysis

# 📊 Tweet Sentiment Analysis using TextBlob

## DESCRIPTION :

This project performs sentiment analysis on a collection of tweets using Python and the TextBlob library. It is designed to automatically determine whether the sentiment expressed in each tweet is **positive**, **negative**, or **neutral**. This kind of analysis can help businesses, researchers, and analysts understand public opinion about specific topics, products, services, or trends by analyzing large volumes of text data from social media.

The script loads a CSV file of tweets, cleans and preprocesses the text, performs sentiment analysis using TextBlob, and visualizes the sentiment distribution using a bar chart. Finally, it saves the resulting data with sentiment labels to a new CSV file for further analysis or reporting.

---

## Objectives :

- Preprocess and clean raw tweet data for accurate analysis.
- Perform sentiment classification using TextBlob’s polarity score.
- Visualize the sentiment distribution across all tweets.
- Save the analyzed data to a CSV file for further use.

---

## Dataset :

The dataset used in this project is a CSV file named `tweets.csv`, which contains raw tweet text collected from Twitter. The file is expected to include at least one column:
- `text`: The content of the tweet.

You can replace the sample dataset with any other Twitter dataset containing a `text` column to use the same analysis pipeline.

---

## Dependencies :

The following Python libraries are required to run this project:

- `pandas`: For data loading and manipulation
- `re`: For regular expressions used in text cleaning
- `string`: For handling punctuation
- `textblob`: For sentiment analysis
- `matplotlib`: For data visualization

pip install pandas textblob matplotlib
---

# WORKING PRINCIPLE :
## 1. Load the Data

 The script begins by loading a CSV file containing tweets from a specified file path.

## 2. Clean the Text

 Tweet texts are cleaned using regular expressions and basic string manipulation. The cleaning steps include:

Removing URLs

Removing mentions (e.g., @username) and hashtags

Removing punctuation

Converting text to lowercase

 This step helps reduce noise in the data and improve the accuracy of sentiment detection.


## 3. Sentiment Analysis

Each cleaned tweet is analyzed using TextBlob, which returns a polarity score:

A score > 0 indicates a positive sentiment.

A score < 0 indicates a negative sentiment.

A score of 0 indicates a neutral sentiment.

The sentiment label is stored in a new column in the DataFrame called sentiment.


## 4. Visualization

 A bar plot is generated to show the number of tweets in each sentiment category. The bars are color-coded:

Green for positive

Red for negative

Blue for neutral

This provides a quick and intuitive way to understand the general mood of the tweets.

## 5. Save the Results

The updated DataFrame with cleaned text and sentiment labels is saved to a new CSV file at data/tweets_with_sentiment.csv



# OUTPUTS :
