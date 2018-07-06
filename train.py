#Import required modules:
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import csv

def import_data(filename):

    all_data = np.genfromtxt(filename,delimiter = ',',dtype = object)
    np.random.shuffle(all_data)
    return all_data

def segregate_data(data):
    
    x,y = np.shape(data)
    labels = data[:,0]
    inputs = data
    inputs = np.delete(inputs, 0, 1)  
    return labels,inputs

def build_nn(x_train,x_test,y_train, y_test):

    labels = convert_to_int(y_train)
    test_labels = convert_to_int(y_test)    

    model = Sequential()
    model.add(Dense(30, input_dim=16))
    model.add(Activation('sigmoid'))
    model.add(Dense(26))
    model.add(Activation('softmax'))
    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    rms = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)

    model.compile(optimizer=rms,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
 
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=26)
    ohl_test = keras.utils.to_categorical(test_labels, num_classes=26)
    
    model.fit(x_train, one_hot_labels, epochs=25, batch_size=32)
    score = model.evaluate(x_test, ohl_test, batch_size=32)
    print(score)
    return model

def convert_to_int(labels):
    '''
    Converts the character labels into integers so
    that they can then be one hot encoded when we try to 
    fit our model.
    
    '''
    labels = np.array(labels, dtype = object)

    for i,label in enumerate(labels):
        labels[i] = ord(label) - ord('A')

    return labels


def train_test(labels,inputs):

    x_train = inputs[:15999]
    x_test = inputs[15999:]
    y_train = np.delete(labels, np.s_[15999:])
    y_test = np.delete(labels, np.s_[0:15999])

    return x_train,x_test,y_train,y_test

def main():

    filename = 'letter-recognition.csv'
    data = import_data(filename)

    labels,inputs = segregate_data(data)
    X_train, X_test, Y_train, Y_test = train_test(labels, inputs)
    print(np.shape(Y_train))
    print(np.shape(X_train))
    print(np.shape(Y_test))
    print(np.shape(X_test))

    model = build_nn(X_train,X_test,Y_train,Y_test)
    model.save('model_84.h5')
    
if __name__=="__main__":
    main()