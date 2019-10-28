import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import re

np.random.seed(500)
Corpus = ""
f = open(r"C:\Users\Admin\PycharmProjects\raw_data\vietnamese_div\iphone8_0.txt", mode='r', encoding='utf-8')
for i in f:
    Corpus += i
f.close()

Corpus = re.sub(' +', ' ', Corpus)

Corpus = [entry.lower() for entry in Corpus]
Corpus = "".join(Corpus)

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(Corpus):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('vietnamese') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
    print(Final_words)

print(Corpus)

