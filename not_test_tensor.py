from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
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
        if i == 'nat':
            lab = 0
        elif i == 'pos':
            lab = 1
        else:
            lab = 2
        data_y.append(lab)

    f_x.close()
    f_y.close()
    f_m.close()



read_data()

train_x = np.array(data_x[0: 1300])
train_y = np.array(data_y[0: 1300])
test_x = np.array(data_x[1300:])
test_y = np.array(data_y[1300:])

print(train_x)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1913,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=10)
test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)

print('\nTest accuracy:', test_acc)

model.save()

print(tf.__version__)