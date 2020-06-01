import tensorflow.keras.backend as K
import numpy as np
import sys
import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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

def load_data_csv(file_name):
    return pd.read_csv(file_name, index_col=0)

def shift_mse(y_true, y_pred):
    loss = (1 + 1/ (1 + K.exp(-(y_pred - y_true)))) * K.square(y_pred - y_true)
    loss = K.mean(loss, axis = 1)
    return loss

def train():
    
    #train_data = load_data("Yasmin_40360_50knn_Trans.csv")

    #X_train = train_data[train_data.columns[:-1]].values
    #Y_train = train_data[train_data.columns[-1]].values
    
    (X_train, Y_train) = load_data_2("All_Data.txt")
    
    model = XGBClassifier(verbose_eval=True, tree_method='gpu_hist', 
                          learning_rate=0.1, max_depth=6, min_child_weight=3, 
                          n_estimators=200, objective='mse_shift', 
                          subsample=1, verbosity=2, silent=0)

    model.fit(X_train, Y_train)
    
    return model

def xgboost_heuristic_2(board, model):

    """
    This function takes in a board and a trained NN model and returns
    the heuristic the model predicts.
    """

    return model.predict(get_rep_2(board).reshape(1,288))
