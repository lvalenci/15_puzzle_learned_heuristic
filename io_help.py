"""
io_help.py
worked on by: Luka Valencic
"""
import numpy as np
from constants import *

def board_to_string(board):
    """converts board to string for printing"""
    flat = (board.reshape(SIZE ** 2)).tolist()
    out = '_'.join([str(f) for f in flat])
    return out

def string_to_board(string):
    """converts board in string represenation output by board_to_string
    to a standard board representation"""
    lst = [int(elem) for elem in string.split('_')]
    ret_val = (np.array(lst)).reshape((SIZE, SIZE))
    return ret_val

def board_and_dist_to_string(board, dist):
    """converts board and distance to standard represtation for printing"""
    board_rep = board_to_string(board)
    dist_rep = str(dist) + '!'
    return dist_rep + board_rep

def string_to_board_and_dist(string):
    """given string output by board_and_dist_to_string, returns encoded
    board and distance, output is tuple (board, dist)"""
    split = string.split('!')
    dist = int(float(split[0]))
    board = string_to_board(split[1])
    return (board, dist)

def test_info_to_string(board, states, time, sol_len):
    """
    given a board, number of states, time, and lenght of solution, converts it
    to standard form for output.
    """
    board_rep = board_to_string(board) + "!"
    state_rep = str(states) + "!"
    time_rep = str(time) + "!"
    len_rep = str(sol_len)
    return board_rep + state_rep + time_rep + len_rep

def string_to_test_info(string):
    """
    given a string containing the standard form of test info, returns tuple of 
    board, number of states to solution, time, and lenght of solution
    """
    split = string.split("!")
    board = string_to_board(split[0])
    n_states = int(split[1])
    time = float(split[2])
    sol_len = int(split[3])
    return (board, n_states, time, sol_len)