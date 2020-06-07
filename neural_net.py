"""
neural_net.py
worked on by: Yasmin Veys and Luka Valencic

Run neural_net.py <training_file_input> 
EX: python3 neural_net.py Meena_5_16_89475.txt saved_model.txt
"""

import numpy as np
import sys
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, Input
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import load_model
import tensorflow.keras.losses

from constants import * 
from heuristic import *
from io_help import *
from solver import *

# Must define all custom loss function here
def custom_loss(y_true, y_pred): 
    """
    This custom loss function takes in y_true and y_pred and returns
    the loss. It cannot take more than two arguments.
    """

    # Slighly different verson of mse, for example 
    loss = K.square((y_pred - y_true)/10)
    loss = K.mean(loss, axis=1)

    return loss

def exp_loss(y_true, y_pred):
    """
    Custom loss function. 
    """
    loss = K.exp((y_pred - y_true)) / 10
    loss = loss + K.square((y_pred - y_true) / 2)
    loss = K.mean(loss, axis = 1)

    return loss

def exp_loss_2(y_true, y_pred):
    """
    Custom loss function. 
    """
    loss = K.exp((y_pred - y_true)) / 2
    loss = loss + K.square(y_pred - y_true)
    loss = K.mean(loss, axis = 1)

    return loss

def shift_mse(y_true, y_pred):
    """custom loss functions"""
    loss = (1 + 1/ (1 + K.exp(-(y_pred - y_true)))) * K.square(y_pred - y_true)
    loss = K.mean(loss, axis = 1)
    return loss

def bounded_loss(input_layer):
    """returns exp loss if prediction greater than actual or less than manhattan, otherwise
    linear loss"""
    def bounded_inside(y_true, y_pred):
        #i2 = K.eval(input_layer)
        board = unencode_board(input_layer)
        man_dist = manhattan(board)
        if (K.eval(y_pred) > K.eval(y_true)):
            return K.exp((y_pred - y_true))
        if (K.eval(y_pred) < man_dist):
            return K.exp((y_true - y_pred))
        return (K.eval(y_true - y_pred))
    return bounded_inside

# define custom loss functions in keras so can load model
tensorflow.keras.losses.custom_loss = custom_loss
tensorflow.keras.losses.exp_loss = exp_loss
tensorflow.keras.losses.exp_loss_2 = exp_loss_2
tensorflow.keras.losses.shift_mse = shift_mse
tensorflow.keras.losses.bounded_loss = bounded_loss


def neural_net_heuristic(board, model):

    """
    This function takes in a board and a trained NN model and returns
    the heuristic the model predicts.
    """
    [[pred]] = model.predict(one_hot_encode(board).reshape(1,256))
    return round(pred)

def neural_net_heuristic_2(board, model):
    """
    This function takes in a board and a trained NN model and returns
    the heuristic the model predicts.
    """
    [[pred]] = model.predict(get_rep_2(board).reshape(1,288))
    return round(pred)

def neural_net_heuristic_3(board, model):
    """
    This function takes in a board and a trained NN model and returns
    the heuristic the model predicts.
    """
    [[pred]] = model.predict(get_rep_3(board).reshape(1,290))
    return round(pred)

def unencode_board(encoding):
    """
    given a one-hot encoding of the board as returned by one_hot_encode,
    returns encoding of board in original format
    """
    s2 = SIZE ** 2
    board = np.zeros(s2)

    for i in range(0,s2):
        for j in range(0,s2):
            curr_ind = i * s2 + j
            if K.get_value(encoding[curr_ind]) == 1:
                board[j] = i + 1
    board = board.reshape((SIZE, SIZE))
    return board

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

def calc_displacements(board):
    """given a board, returns SIZE^2 array containing distances of tile in
    each entry to proper location"""
    dis_x = np.zeros(SIZE ** 2)
    dis_y = np.zeros(SIZE ** 2)

    for i in range(SIZE):
        for j in range(SIZE):
            curr = board[i,j]
            (x, y) = get_proper_loc(curr)
            dis_x[SIZE * i + j] = x-i
            dis_y[SIZE * i + j] = y-j
    return np.concatenate((dis_x, dis_y))

def load_data(file_name):
    """
    This function reads in training data from a file and returns 
    the one-hot encoded data X and their labels Y as a tuple. 
    """
    file = open(file_name, "r")

    X = []
    Y = []

    for string in file: 
        (board, dist) = string_to_board_and_dist(string) 

        X.append(one_hot_encode(board))
        Y.append(dist)

    file.close()

    return(np.asarray(X),np.asarray(Y))

def get_rep_2(board):
    """returns representation of one-hot encoded board with additional 16 
    entries which encode distnaces entry in eqch square is from proper location"""
    encode = one_hot_encode(board)
    displacements = calc_displacements(board)
    return np.concatenate((encode, displacements))

def load_data_2(file_name):
    """same as load_data except that has additional 16 entries which
    encode distnaces entry in eqch square is from proper location"""
    file = open(file_name, "r")

    X = []
    Y = []
    for line in file:
        (board, dist) = string_to_board_and_dist(line)
        Y.append(dist)
        X.append(get_rep_2(board))


    file.close()
    return (np.asarray(X), np.asarray(Y))

def get_rep_3(board):
    """returns representation of one-hot encoded board with additional 16 
    entries which encode distnaces entry in eqch square is from proper location
    also has manhattan and hamming metrics
    """
    encode = one_hot_encode(board)
    displacements = calc_displacements(board)
    man = np.asarray([manhattan(board, None), hamming(board, None)])

    return np.concatenate((encode, displacements, man))

def load_data_3(file_name):
    """same as load_data except that has additional 16 entries which
    encode distnaces entry in eqch square is from proper location"""
    file = open(file_name, "r")

    X = []
    Y = []
    for line in file:
        (board, dist) = string_to_board_and_dist(line)
        Y.append(dist)
        X.append(get_rep_3(board))

    file.close()
    return (np.asarray(X), np.asarray(Y))

def find_over_estimate(file_name, model):
    """
    This function takes in a model saved in model_file and data points in 
    file_name and prints out the percentage of times said model predicted 
    a distance greater than the actual distance and the percentage of times
    said model predicted a distance less than the Manhattan Distance
    """
    data = open(file_name, "r")
    over = []
    under = []

    for line in data:
        (board, dist) = string_to_board_and_dist(line)
        man_dist = manhattan(board, None)
        pred = neural_net_heuristic_3(board, model)
        over.append(pred > dist)
        under.append(pred < man_dist)

    print("prediction less than manhattan percent of the time", sum(under) * 100 / len(under))
    print("prediction greater than actual distance precent of the time", sum(over) * 100 / len(over))


def evaluate(file_name):
    """
    This function reads in training data from a file and 
    trains and evaluates NN model using kfold validation. 
    """
    (X,Y) = load_data(file_name)

    # Implement K-fold cross validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=2020)

    for train, test in kfold.split(X, Y):
        # Build Model
        model = Sequential()

        # Input Layer
        model.add(Dense(units=256, input_dim=256, activation='relu'))
        model.add(Dropout(0.1))
        # Hidden Layers
        model.add(Dense(units=256, activation='relu'))
        # Output Layer
        model.add(Dense(units=1, activation='linear'))

        # Define the optimizer and loss function
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # You can also define a custom loss function
        # model.compile(optimizer='adam', loss=custom_loss)

        # Train 
        model.fit(X[train], Y[train], epochs=20)

        # Evaluate
        score = model.evaluate(X[test], Y[test], verbose=0)
        print(score)

    return model


def evaluate_custom_funcs(file_name, cust_loss, cust_metric):
    """
    This function reads in training data from a file and 
    trains and evaluates NN model using kfold validation. 
    Specifically uses custom loss function with 
    """
    (X,Y) = load_data(file_name)

    # Implement K-fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=2020)

    for train, test in kfold.split(X, Y):
        # Build Model
        i = Input(shape = (256,))
        x_1 = Dense(256, activation='relu')(i)
        x_2 = Dropout(0.1)(x_1)
        x_3 = Dense(64, activation='relu')(x_2)
        x_4 = Dropout(0.1)(x_3)
        x_5 = Dense(16, activation='relu')(x_4)
        o = Dense(1, activation='linear')(x_1)
        model = Model(i,o)

        # Define the optimizer and loss function
        # model.compile(optimizer='adam', loss=cust_loss, metrics=cust_metric(i))
        model.compile(optimizer = 'adam', loss = cust_loss, metrics=['accuracy'])
        # You can also define a custom loss function
        # model.compile(optimizer='adam', loss=custom_loss)

        # Train 
        model.fit(X[train], Y[train], epochs=15)

        # Evaluate
        score = model.evaluate(X[test], Y[test], verbose=0)
        print(score)

    return model

def evaluate_data_2(file_name, cust_loss, cust_metric):
    """
    This function reads in training data from a file and 
    trains and evaluates NN model using kfold validation. 
    Specifically uses custom loss function with 
    """
    (X,Y) = load_data_2(file_name)

    # Implement K-fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=2020)

    for train, test in kfold.split(X, Y):
        # Build Model
        i = Input(shape = (288,))
        x_1 = Dense(288, activation='relu')(i)
        x_2 = Dropout(0.1)(x_1)
        x_3 = Dense(64, activation='relu')(x_2)
        x_4 = Dropout(0.1)(x_3)
        x_5 = Dense(16, activation='relu')(x_4)
        o = Dense(1, activation='linear')(x_1)
        model = Model(i,o)

        # Define the optimizer and loss function
        # model.compile(optimizer='adam', loss=cust_loss, metrics=cust_metric(i))
        model.compile(optimizer = 'adam', loss = cust_loss, metrics=['accuracy'])
        # You can also define a custom loss function
        # model.compile(optimizer='adam', loss=custom_loss)

        # Train 
        model.fit(X[train], Y[train], epochs=15)

        # Evaluate
        score = model.evaluate(X[test], Y[test], verbose=0)
        print(score)

    return model

def train(file_name):
    """
    This function reads in training data from a file and returns a 
    trained NN model. 
    """
    (X,Y) = load_data(file_name)

    # Build Model
    model = Sequential()

    # Input Layer
    i = Input(shape = (256,))
    x_1 = Dense(256, activation='relu')(i)
    x_2 = Dropout(0.1)(x_1)
    x_3 = Dense(64, activation='relu')(x_2)
    x_4 = Dropout(0.1)(x_3)
    x_5 = Dense(16, activation='relu')(x_4)
    o = Dense(1, activation='linear')(x_1)
    model = Model(i,o)

    # Define the optimizer and loss function
    model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])

    # You can also define a custom loss function
    # model.compile(optimizer='adam', loss=custom_loss)

    # Train 
    model.fit(X, Y, epochs=15)

    return model

def train_custom_loss(file_name, loss_func):
    """
    This function reads in training data from a file and returns a 
    trained NN model.  This NN model is trained with specificed loss
    function
    """
    (X,Y) = load_data(file_name)

    # Build Model
    model = Sequential()

    # Input Layer
    i = Input(shape = (256,))
    x_1 = Dense(256, activation='relu')(i)
    x_2 = Dropout(0.1)(x_1)
    x_3 = Dense(64, activation='relu')(x_2)
    x_4 = Dropout(0.1)(x_3)
    x_5 = Dense(16, activation='relu')(x_4)
    o = Dense(1, activation='linear')(x_1)
    model = Model(i,o)

    # Define the optimizer and loss function
    model.compile(optimizer='adam', loss=loss_func, metrics=['accuracy'])

    # You can also define a custom loss function
    # model.compile(optimizer='adam', loss=custom_loss)

    # Train 
    model.fit(X, Y, epochs=15)

    return model

def train_custom_loss_2(file_name, loss_func):
    """
    This function reads in training data from a file and returns a 
    trained NN model.  This NN model is trained with specificed loss
    function
    """
    (X,Y) = load_data_2(file_name)

    # Build Model
    model = Sequential()

    # Input Layer
    i = Input(shape = (288,))
    x_1 = Dense(288, activation='relu')(i)
    x_2 = Dropout(0.1)(x_1)
    x_3 = Dense(64, activation='relu')(x_2)
    x_4 = Dropout(0.1)(x_3)
    x_5 = Dense(16, activation='relu')(x_4)
    o = Dense(1, activation='linear')(x_1)
    model = Model(i,o)

    # Define the optimizer and loss function
    model.compile(optimizer='adam', loss=loss_func, metrics=['accuracy'])

    # You can also define a custom loss function
    # model.compile(optimizer='adam', loss=custom_loss)

    # Train 
    model.fit(X, Y, epochs=15)

    return model

def train_custom_loss_3(file_name, loss_func):
    """
    This function reads in training data from a file and returns a 
    trained NN model.  This NN model is trained with specificed loss
    function
    """
    (X,Y) = load_data_3(file_name)

    # Build Model
    model = Sequential()

    # Input Layer
    i = Input(shape = (290,))
    x_1 = Dense(290, activation='relu')(i)
    x_2 = Dropout(0.1)(x_1)
    x_3 = Dense(64, activation='relu')(x_2)
    x_4 = Dropout(0.1)(x_3)
    x_5 = Dense(16, activation='relu')(x_4)
    o = Dense(1, activation='linear')(x_1)
    model = Model(i,o)

    # Define the optimizer and loss function
    model.compile(optimizer='adam', loss=loss_func, metrics=['accuracy'])

    # You can also define a custom loss function
    # model.compile(optimizer='adam', loss=custom_loss)

    # Train 
    model.fit(X, Y, epochs=15)

    return model

def run_saved_model(model_file, data_file):
    """
    given a file to which a model is saved a a datafile, runs model on data
    on code to evaluate accuracy and score
    """
    model = load_model(model_file)
    (X, Y) = load_data(data_file)
    score = model.evaluate(X, Y, verbose = 0)
    print(score)

if __name__ == "__main__":

    if not (len(sys.argv) == 3):
        print("usage error:\n arg1 = input file name \n arg2 = file to save model to")
        exit()

    file_name = sys.argv[1] 
    out_file = sys.argv[2]

    # Toy Example Testing 
    # To train on the entire data set, replace evaluate with train
    #model = evaluate(file_name)
    #model = evaluate_custom_funcs(file_name, exp_loss, None)
    model = train_custom_loss_2(file_name, exp_loss_2)
    #model.save(out_file)
    board = gen_board()
    print(neural_net_heuristic_2(board, model))
    print(manhattan(board, None))
    find_over_estimate(file_name, model)