import pandas as pd
import spacy
import numpy as np
from termcolor import colored
import warnings
warnings.filterwarnings(action='ignore')
fake_df = pd.read_csv("./Fake.csv")
true_df = pd.read_csv("./True.csv")

fake_df['Label'] = "Fake"
true_df['Label'] = "True"

df = pd.concat([fake_df, true_df], axis=0)
df.head()


# print("Shape of fake_df: " , fake_df.shape)
# print("Shape of true_df: " ,true_df.shape)
# print("Shape of df: " ,df.shape)


df = df.drop(['title', 'subject', 'date'], axis=1)
df.head()
print(colored("\n'TITLE','DATE' AND 'SUBJECT' COLUMNS WERE SUCCESFULLY DROPPED...", "green"))

print(df['Label'].value_counts())

df['label_encode'] = df['Label'].map({'Fake':0, 'True':1})
df.head()
print(colored("\nDATASETS WERE SUCCESFULLY MERGED...", "green"))

# drop duplicated values from the dataset

df.drop_duplicates(inplace = True)

print(colored("\nDUPLICATED VALUES WERE SUCCESFULLY DROPPED...", "green"))

#convert uppercase letters to lowercase letters

df["text"] = df["text"].apply(lambda x: " ".join(x.lower() for x in x.split()))

print(colored("\nCONVERTED SUCCESFULLY...", "green"))

#delete punctuation marks

df["text"] = df["text"].str.replace('[^\w\s]','')

print(colored("\nDELETED PUNCTUATION MARKS SUCCESFULLY...", "green"))


#delete numbers

df["text"] = df["text"].str.replace('\d','')

print(colored("\n NUMBERS DELETED SUCCESFULLY...", "green"))


import re, string, unicodedata
from string import punctuation
import nltk
from nltk.corpus import stopwords
from textblob import Word
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS

#delete stopwords

stop_words = set(stopwords.words("english"))
punctuation = list(string.punctuation)
stop_words.update(punctuation)

df["text"] = df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

print(colored("\nSTOPWORDS DELETED SUCCESFULLY...", "green"))

#lemmatization. That is, we get the roots of the words

df["text"] = df["text"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

print(colored("\nLEMMATIZED SUCCESFULLY...", "green"))

from datetime import datetime

from spellchecker import SpellChecker

# Create a SpellChecker object
spell = SpellChecker()
# Feature Extraction
df['words'] = df.text.apply(lambda x:re.findall(r'\w+', x ))
df['errors'] = df.words.apply(spell.unknown)
df['errors_count'] = df.errors.apply(len)
df['words_count'] = df.words.apply(len)
df['sentence_length'] = df.text.apply(len)

def label_sentiment(x:float):
    if x < -0.05 : return 'negative'
    if x > 0.35 : return 'positive'
    return 'neutral'
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from tqdm.notebook import tqdm
import nltk
nltk.download('vader_lexicon')
sia = SIA() 
# Extract Sentiment Values for each tweet 
df['sentiment'] = [sia.polarity_scores(x)['compound'] for x in tqdm(df['text'])]
df['overall_sentiment'] = df['sentiment'].apply(label_sentiment);
