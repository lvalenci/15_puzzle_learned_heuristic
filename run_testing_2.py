"""
worked on by: Luka
runs program to test learned model versus manhattan heuristic
currently implemented to use neural_net, only small tweaks required
for other models
"""
import sys
import numpy as np

from constants import *
import heuristic as h
import io_help as io
import neural_net as nn
import xg_boost as xg
import xg_boost_2 as xg2
import solver as s
import pickle


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

def run_testing(data_file, model, h_func):
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

    for i in range(len(boards)):

        print("{}/{}".format(i+1,len(boards)))

        (c_states, c_time, sol_path) = s.solve(boards[i], h_func, model)

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

    print("about to train")
    model = pickle.load(open("xg_model_mse_shift_fe_slow", "rb"))
    #model = nn.train_custom_loss_2("All_Data.txt", nn.shift_mse)
    print("finished training")
    #h_func = nn.neural_net_heuristic_2

    run_testing("Test_boards.txt", model, xg2.xgboost_heuristic_2)

    
