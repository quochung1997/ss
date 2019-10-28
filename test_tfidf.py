docA = "the quick brown fox jumps over the lazy dog and"
docB = "never jump over the lazy dog quickly"

bowA = docA.split(" ")
bowB = docB.split(" ")

f = open('iphone8_0.txt', 'r')
iphone = ""

for i in f:
    iphone += i

iphone.replace('\n', ' ')
import re
iphone = re.sub(' +', ' ', iphone)
word_list_iphone = iphone.split(' ')

#Create dictionary
word_dict = set(bowA).union(set(bowB))

wordDictA = dict.fromkeys(word_dict, 0)
wordDictB = dict.fromkeys(word_dict, 0)

#count the word in bads
for word in bowA:
    wordDictA[word]+=1

for word in bowB:
    wordDictB[word]+=1

print(wordDictA)
print(wordDictB)


# calculate TF
def compute_TF(word_dict, bow):
    tf_dict = {}
    bow_count = len(bow)
    for word in word_dict:
        tf_dict[word] = word_dict[word] / float(bow_count)

    return tf_dict


tf_bowA = compute_TF(wordDictA, bowA)
tf_bowB = compute_TF(wordDictB, bowB)

print(tf_bowA)
print(tf_bowB)


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


idfs = compute_IDF([wordDictA, wordDictB])

print(idfs)


def compute_TFIDF(tf_bow, idfs):
    tfidf = {}
    for word in tf_bow:
        tfidf[word] = tf_bow[word]*idfs[word]
    return tfidf

tfidf_bowA = compute_TFIDF(tf_bowA, idfs)
tfidf_bowB = compute_TFIDF(tf_bowB, idfs)

import pandas as pd

df = pd.DataFrame([tfidf_bowA, tf_bowB])
print(df)
