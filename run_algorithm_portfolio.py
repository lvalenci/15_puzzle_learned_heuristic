import sys
import numpy as np

from constants import *
from tensorflow.keras.models import load_model
import keras.losses
import heuristic as h
import io_help as io
from neural_net import *
import xg_boost as xg
import xg_boost_2 as xg2
import solver as s
import pickle

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

def load_boards(filename):
    """
    given name of file containing test boards, loads all test boards
    """
    file = open(filename, "r")

    boards = []
    n_states = []
    times = []
    dists = []

    for line in file:
        (board, c_states, c_time, sol_len) = io.string_to_test_info(line)
        boards.append(board)
        n_states.append(c_states)
        times.append(c_time)
        dists.append(sol_len)

    return (boards, n_states, times, dists)

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

def heuristic_gen(all_models, all_heuristics, alg_model):
    def portfolio_heuristic(board, trash):
        i = np.asarray(alg_model.predict(one_hot_encode(board).reshape(1,256)).flatten()).argmax()
        if (i == 4):
            i = 3
        pred = all_heuristics[i](board, all_models[i])
        return pred
    return portfolio_heuristic


def run_testing(data_file, alg_model):
    """
    given a data_file containing testing data, a model, and heuristic function
    for said model, computes average number of states to solution, number to 
    times solution length is non-optimal, and average estimates of solution 
    lengths
    """
    (boards, n_states, times, dists) = load_boards(data_file)

    cust_states = []
    cust_wrong = 0
    cust_distance = []

    model_0 = load_model("nn_shift_mse_rep_2.txt", compile=False)
    model_1 = load_model("nn_shift_mse_rep_3.txt", compile=False)
    model_2 = pickle.load(open("./XGBoost_Models/xg_model_mse_shift_fe_500", "rb"))

    all_models = [model_0, model_1, model_2, None]
    all_heuristics = [neural_net_heuristic_2, neural_net_heuristic_3, xg2.xgboost_heuristic_2, manhattan]

    for i in range(len(boards)):

        print("{}/{}".format(i+1,len(boards)))

        (c_states, c_time, sol_path) = s.solve(boards[i], heuristic_gen(all_models, all_heuristics, alg_model), None)

        cust_states.append(c_states)
        sol_len = len(sol_path) - 1
        print("c_states: {},sol_len: {}".format(c_states, sol_len))
        if not (sol_len == dists[i]):
            cust_wrong += 1
        cust_distance.append(sol_len)

    print("average number of states explored to find solution:")
    print("\tfor learned model: " + str(np.mean(cust_states)))
    print("\tfor manhattan distance: " + str(np.mean(n_states)))
    print("----------------------------------------------------")
    print("solution was non-optimal " + str(cust_wrong / NUM_TEST_BOARDS * 100) + "% of the time")
    print("----------------------------------------------------")
    print("average length of solution path was:")
    print("\tfor learned model: " + str(np.mean(cust_distance)))
    print("\tfor manhattan distance: " + str(np.mean(dists)))


if __name__ == "__main__":

    # define custom loss functions in keras so can load model
    keras.losses.custom_loss = custom_loss
    keras.losses.exp_loss = exp_loss
    keras.losses.exp_loss_2 = exp_loss_2
    keras.losses.shift_mse = shift_mse
    keras.losses.bounded_loss = bounded_loss

    print("Loading algorithm model...")
    algorithm_model = load_model("algorithm_model_50_no_do.h5")

    run_testing("Test_boards.txt", algorithm_model)

    
