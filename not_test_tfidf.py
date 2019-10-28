import re
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

# calculate TF
def compute_TF(word_dict, bow):
    tf_dict = {}
    bow_count = len(bow)
    for word in word_dict:
        tf_dict[word] = word_dict[word] / (float(bow_count))

    return tf_dict


def compute_IDF(doc_list):
    import math
    idf_dict = {}
    N = len(doc_list)

    # count number of documents that contain this word
    idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
    for doc in doc_list:
        for word in doc:
            if doc[word] > 0:
                idf_dict[word] += 1

    for word in idf_dict:
        idf_dict[word] = math.log(N / float(idf_dict[word]))

    return idf_dict


def compute_TFIDF(tf_bow, idfs):
    tfidf = {}
    for word in tf_bow:
        tfidf[word] = tf_bow[word]*idfs[word]
    return tfidf


def check_if_number(s):
    return '0' in s or '1' in s or '2' in s or '3' in s or '4' in s or '5' in s or '6' in s or '7' in s or '8' in s or '9' in s


class DataWord():
    def __init__(self):
        self.bow = ""
        self.word_dict = {}
        self.tf_bow = {}
        self.idfs = {}
        self.tfidf = {}
        self.label = ""


un_f = open('unnecessary_words', 'r', encoding='utf-8')
unnes_word = set()

for line in un_f:
    words = line.split()
    for word in words:
        if word.isalpha():
            unnes_word.add(word)

un_f.close()
filename_nat = 'Nat'
filename_neg = 'Neg'
filename_pos = 'Pos'
word_set = set()

data_words = []


def read_file(label, fn):
    f = open(fn+'.txt', 'r', encoding='utf-8')
    global data_words
    for line in f:
        dw = DataWord()
        line = line.lower()
        line = line.replace('\n', ' ')
        arr = line.split(' ')
        new_arr = []
        for a in arr:
            if a.isalpha():
                new_arr.append(a)
                word_set.add(a)
        dw.bow = new_arr
        dw.label = label
        print(dw.bow)
        if len(dw.bow) != 0:
            data_words.append(dw)

    f.close()


read_file('nat', filename_nat)
read_file('neg', filename_neg)
read_file('pos', filename_pos)
word_dict_all = []

for dw in data_words:
    dw.word_dict = dict.fromkeys(word_set, 0)
    for word in dw.bow:
        dw.word_dict[word] += 1
    dw.tf_bow = compute_TF(dw.word_dict, dw.bow)
    word_dict_all.append(dw.word_dict)

idfs = compute_IDF(word_dict_all)
f_idfs = open('idfs_2.txt', 'w', encoding='utf-8')
for i in idfs:
    f_idfs.write(i+" "+str(idfs[i])+'\n')

random.shuffle(data_words)
data = []
data_y = []
for dw in data_words:
    dw.tfidf = compute_TFIDF(dw.tf_bow, idfs)
    print(dw.tfidf)
    data.append(dw.tfidf)
    data_y.append(dw.label)


f_save_trainx = open('train_x.txt', 'w', encoding='utf-8')
f_save_trainy = open('train_y.txt', 'w', encoding='utf-8')
f_save_metrix = open('metrix.txt', 'w', encoding='utf-8')

for d in data:
    s = ""
    for i in d:
        s += str(d[i])+" "
    f_save_trainx.write(s+'\n')

for d in data_y:
    # print(d)
    f_save_trainy.write(d+'\n')

for w in word_set:
    f_save_metrix.write(w+'\n')

f_save_metrix.close()
f_save_trainx.close()
f_save_trainy.close()

df = pd.DataFrame(data)
print(df)

print(len(word_set))
print(word_set)