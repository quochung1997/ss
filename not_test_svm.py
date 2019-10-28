from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import numpy as np
import random

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

def read_data():
    f_x = open('train_x.txt', 'r', encoding='utf-8')
    f_y = open('train_y.txt', 'r', encoding='utf-8')
    f_m = open('metrix.txt', 'r', encoding='utf-8')

    global data_x, data_y, data_m
    data_m = set()
    data_x = []
    data_y = []
    for i in f_m:
        data_m.add(i)
    for i in f_x:
        arr = i.split(' ')
        # print(arr)
        arr_fl = []
        for a in arr:
            if a != '\n':
                arr_fl.append(float(a))
        data_x.append(arr_fl)
    for i in f_y:
        if i == 'nat\n':
            lab = 0
        elif i == 'pos\n':
            lab = 1
        else:
            lab = 2
        data_y.append(lab)

    f_x.close()
    f_y.close()
    f_m.close()



read_data()
n = 2500
train_x = np.array(data_x[0: n])
train_y = np.array(data_y[0: n])
test_x = np.array(data_x[n:])
test_y = np.array(data_y[n:])


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(train_x, train_y)


# predict the labels on validation dataset
predictions_SVM = SVM.predict(test_x)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, test_y)*100)

import pickle


# Classifier - Algorithm - Naive Bayes
# fit the training dataset on the classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(train_x, train_y)

# predict the labels on validation dataset
predictions_NB = Naive.predict(test_x)

# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, test_y)*100)

pickle.dump(Naive, open('finalized_model_naivie_2.sav', 'wb'))
pickle.dump(SVM, open('finalized_model_svm_2.sav', 'wb'))