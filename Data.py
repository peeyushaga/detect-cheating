import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
from sklearn.utils import shuffle


def read_data(path, value):
    sentences = []
    targets = []

    with open(path, 'r') as f:
        for line in f:
            sentences.append(line.strip())
            targets.append(value)
    df = pd.DataFrame({'sentences': sentences, 'targets': targets})
    return df


def clean_text(text):
    text = text.lower()
    text = text.strip()
    text=re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)

    return text

def stop_words(text):
    return ' '.join(word for word in text.split() if word.lower() not in stopwords.words('english'))

stemmer = SnowballStemmer('english')
def snow_stemmer(text):
  return ' '.join(stemmer.stem(word) for word in word_tokenize(text))

lemmatizer = WordNetLemmatizer()
def wordnet_lemmatizer(text):
  return ' '.join(lemmatizer.lemmatize(word) for word in word_tokenize(text))

def preprocess(text):
  return wordnet_lemmatizer(snow_stemmer(stop_words(clean_text(text))))

