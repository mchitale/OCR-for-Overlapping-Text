'''OCR of a dataset of handwritten characters, cut into half

Author: Maitreyi Chitale
Date: 29-06-2018

Observations: Using SGD, LR = 0.05, #Epochs = 22,
Test Accuracy = 74.05%, Shuffled Data.

'''

#Import required modules:
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

def data_import(filename, flag):
    '''
    This function takes a filename (of format tsv/csv/xlsx)
    and imports the data into a numpy array.
    It then deletes the columns that we aren't using.
    It returns a shuffled pandas frame of the relevant data.

    '''
    all_data = np.genfromtxt(filename, skip_header = True, dtype = object)
    #convert the array into a pandas frame
    all_data = pd.DataFrame(all_data)
    
    #removing unused columns - 
    del all_data[3] #word_number(unused)
    del all_data[4] #position of letter in word(unused)
    del all_data[5] #cross-validation fold(to split train/test evenly)
    
    #remove the bottom 64 pixels:-
    if flag == '1':    
        for i in range(70,134):
            del all_data[i]
    #remove the top 64 pixels
    elif flag == '2':
        for i in range(6,70):
            del all_data[i]

    #Shuffle data:-
    return all_data.sample(frac=1).reset_index(drop=True)


def segregate_data(data):
    '''
    This function takes the entire pandas dataframe 
    and divides it up into training data, validation data
    and test data. Returns three different pandas dataframes.
    '''
    train_data = data[1:42994]
    test_data = data[42995:]

    return train_data, test_data

def split_labels(data):
    '''
    The function split_labels splits the data into 
    x-y pairs, i.e it separates out the input data
    and their corresponding labels. Returns one pandas
    series(Y) and one pandas dataframe(X). 
    '''
    labels = data[:][1]
    pixel_val = data.copy()
    del pixel_val[0]
    del pixel_val[1]
    del pixel_val[2]

    return pixel_val,labels

def build_nn(data,labels,test_data,test_labels):

    '''
    The function build_nn is responsible for building our
    neural network model, specifying the activation functions,
    number of nodes for every layer, the learning rate, optimizer
    type, type of loss, and metrics to measure our model's performance.
    It then calls the .fit() and .evaluate() function and prints
    out the performance of the model.

    '''
    labels = convert_to_int(labels)
    test_labels = convert_to_int(test_labels)

    model = Sequential()
    model.add(Dense(75, input_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(26))
    model.add(Activation('softmax'))
    
    #Stochastic Gradient Descent - 
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=26)
    ohl_test = keras.utils.to_categorical(test_labels, num_classes=26)

    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
 
    
    model.fit(data, one_hot_labels, epochs=22, batch_size=64)
    score = model.evaluate(test_data, ohl_test, batch_size=64)
    model.save('ocrmodel.h5')
    return score

    
def convert_to_int(labels):
    '''
    Converts the character labels into integers so
    that they can then be one hot encoded when we try to 
    fit our model.
    
    '''
    labels = np.array(labels, dtype = object)

    for i,label in enumerate(labels):
        labels[i] = ord(label) - ord('a')

    return labels

if __name__ == '__main__':
    
    #import all data
    #Data is available in a tsv file that is a flattened
    #array of 0 or 1 (thresholded) values. The array is flattened
    #from a 16x8 image so we have 128 pixel values. 
    #command line argument is a flag that tells us whether to take top 64
    #pixels or bottom 64 pixels.
    flag = sys.argv[1]
    data = data_import("C:/Users/machital/Desktop/letter.tsv", flag)

    #Divide up the data into train, validation & test
    train_data, test_data = segregate_data(data)

    #Split into input and output - 
    train_pixels, train_labels = split_labels(train_data)
    
    test_pixels, test_labels = split_labels(test_data)    
    
    #Validate the shapes of the divided data - 
    print(np.shape(train_pixels))
    print(np.shape(train_labels))

    #Verify the datatype of the divided data-
    print(type(train_pixels))
    print(type(train_labels))

    #Build model and fit it - 
    accuracy = build_nn(train_pixels,train_labels,test_pixels,test_labels)
    
    print('The accuracy of the model is ',round(accuracy[1]*100),'%')