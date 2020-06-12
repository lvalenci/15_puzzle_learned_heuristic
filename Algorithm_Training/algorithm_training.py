import numpy as np
import sys
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, Input
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import load_model
import tensorflow.keras.losses
from keras.utils import to_categorical

from constants import * 
from heuristic import *
from io_help import *
from solver import *

def one_hot_encode(board):
    """ 
    This function one hot encodes the board into a length 256 array.
    The one hot encoding gives the location of each number in the board.
    For example, the first 16 of the 256 numbers will indicate where on
    the board the 1 tile is. 
    """

    flat = (board.reshape(SIZE ** 2)).tolist()

    X = []
    for i in np.arange(1,17): 
        encoding = np.zeros(SIZE ** 2)
        encoding[flat.index(i)] = 1

        X.append(encoding)  

    X = (np.asarray(X).reshape(SIZE ** 4))

    # Potentially append Manhattan distance. 
    # np.append(X, manhattan(board))

    return X

def load_data(file_name):
    """
    This function reads in training data from a file and returns 
    the one-hot encoded data X and their labels Y as a tuple. 
    """
    file = open(file_name, "r")

    X = []
    Y = []

    for string in file: 
        (board, classification) = string_to_board_and_dist(string) 

        X.append(one_hot_encode(board))
        Y.append(classification)

    file.close()

    return(np.asarray(X),np.asarray(Y))

def evaluate(X,Y):
    """
    This function reads in training data from a file and 
    trains and evaluates NN model using kfold validation. 
    """
    #(X,Y) = load_data(file_name)

    #Y = to_categorical(Y)

    # Implement K-fold cross validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=2020)

    for train, test in kfold.split(X, Y):
        # Build Model
        model = Sequential()

        # Input Layer
        i = Input(shape = (256,))
        x_1 = Dense(256, activation='relu')(i)
        x_2 = Dropout(0.2)(x_1)
        o = Dense(5, activation='softmax')(x_2)
        model = Model(i,o)

        # Define the optimizer and loss function
        model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

        # Train 
        model.fit(X, Y, epochs=15, verbose=1)

        # Evaluate
        score = model.evaluate(X[test], Y[test], verbose=0)
        print(score)

    return model

def train(X,Y):
    """
    This function reads in training data from a file and returns a 
    trained NN model. 
    """
    #(X,Y) = load_data(file_name)

    #Y = to_categorical(Y)

    # Build Model
    model = Sequential()

    # Input Layer
    i = Input(shape = (256,))
    x_1 = Dense(256, activation='relu')(i)
    x_2 = Dropout(0.2)(x_1)
    o = Dense(5, activation='softmax')(x_2)
    model = Model(i,o)

    # Define the optimizer and loss function
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    # Train 
    model.fit(X, Y, epochs=50, verbose=1)

    return model

if __name__ == "__main__":

	model = train("final_portfolio_data.txt")

	board = gen_board()
	print(board)
	print(model.predict(one_hot_encode(board).reshape(1,256)))

