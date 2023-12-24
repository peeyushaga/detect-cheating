from Data import read_data, preprocess
import pandas as pd
from sklearn.utils import shuffle
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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer


clean_df = read_data("clean.txt", 0)
cheat_df = read_data("cheat.txt", 1)

df = pd.concat([cheat_df, clean_df], ignore_index=True)
df = shuffle(df)

df['clean_sentences']=df['sentences'].apply(lambda sentence: preprocess(sentence))
X_train, X_val, y_train, y_val = train_test_split(df["clean_sentences"],df["targets"],test_size=0.2, shuffle=True)

tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors = tfidf_vectorizer.fit_transform(X_train)  
X_val_vectors = tfidf_vectorizer.transform(X_val)  

model = LogisticRegression()
model.fit(X_train_vectors, y_train) 

def predict_sentence(sentence):
    sentence = preprocess(sentence)  
    X_vector = tfidf_vectorizer.transform([sentence])  # convert the input sentence to a vector
    y_predict = model.predict(X_vector)  # use the trained model to make a prediction
    y_prob = model.predict_proba(X_vector)[:, 1]  # get the probability of the prediction
    return y_predict, y_prob