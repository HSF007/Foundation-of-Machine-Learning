import pandas as pd
from nltk.stem.porter import PorterStemmer
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


directory_name = os.path.dirname(__file__)
data_file_path = os.path.join(directory_name, 'spam_or_not_spam.csv')
data = pd.read_csv(data_file_path)


def preprocess_text(text):
    stop_words = {'were', 'has', 'out', 'now', 'this', 'above', 'didn', 'until', "couldn't",
              'hasn', 'our', 'been', 'isn', "aren't", 'they', 'again', "it's", 'on', 'yours',
              'down', 'same', "needn't", "didn't", 'doesn', 'whom', 'some', 'as', "that'll",
              "wasn't", 'his', 'just', 'so', 'o', 'her', 'between', "you'll", 'through', 'aren',
              'off', 'then', "mightn't", 'should', 'such', 'hadn', 'those', 'a', 'too', 'are',
              'be', 'do', 'at', 'yourselves', 'no', 'needn', "hadn't", 'ours', 'ourselves', 'before',
              'few', 'any', 'itself', 'there', 'when', 'theirs', 'couldn', 'during', 'having', 'can',
              'own', 'him', 'because', 're', 'ma', 'is', 'into', 'why', 'weren', 'their', 'who', 'y',
              'that', 'my', 'shan', 'by', 'yourself', 'its', 'me', 'does', 'for', "shan't", 'only', 'm',
              'themselves', "you've", 'doing', 'to', 'other', 've', 'd', 'won', "wouldn't", 'had', "isn't",
              'them', 'up', 'or', "don't", 'where', 'your', 'did', "hasn't", 'we', 'both', "you'd", 'than',
              'don', 'mustn', 'ain', "mustn't", 'here', 'mightn', 'i', 'was', 'have', "shouldn't", 'while',
              'and', 'these', 'am', 'not', 'it', 'wasn', "she's", 's', "weren't", 'all', 'being', 'herself',
              'the', 'what', 'she', 'more', 'with', 'about', 'shouldn', 'under', 'how', "should've", "won't",
              'after', 'wouldn', 'further', 'over', 't', 'nor', "you're", 'below', 'in', "doesn't", "haven't",
              'll', 'haven', 'once', 'will', 'against', 'very', 'which', 'himself', 'you', 'he', 'but', 'of',
              'if', 'an', 'myself', 'from', 'hers', 'most', 'each'}
    
    stemmer = PorterStemmer()
    # Lowercase the text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stop words and apply stemming
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)


# Apply preprocessing to each email
data['clean_text'] = data['email'].apply(lambda x: preprocess_text(x) if isinstance(x, str) else "")

data = data[data['clean_text'] != ""]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], data['label'], test_size=0.2, random_state=42)


# Vectorize text data
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train).toarray()
X_test_vect = vectorizer.transform(X_test).toarray()

