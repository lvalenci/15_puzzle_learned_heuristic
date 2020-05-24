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
import solver as s


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
        (c_states, c_time, sol_path) = s.solve(boards[i], h_func, model)
        cust_states.append(c_states)
        sol_len = len(sol_path) - 1
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

    if not (len(sys.argv) == 4):
        print("usage error:\narg1 = file of test boards")
        print("arg2 = saved model file")
        print("arg3 = type of model (nn, cnn, xgboost)")
        exit()

    model = None
    h_func = None
    if (sys.argv[3] == 'nn'):
        model = nn.load_model(sys.argv[2])
        h_func = nn.neural_net_heuristic
    if (sys.argv[3] == 'cnn'):
        exit()
        # TODO
    if (sys.argv[3] == 'xgboost'):
        exit()
        # TODO

    run_testing(sys.argv[1], model, h_func)

    
