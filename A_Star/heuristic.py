"""
heuristic.py
Worked on by: Luka Valencic
"""
import numpy as np
from constants import *

def get_proper_loc(val):
    """
    give an integer value in [0,16], returns the index (i,j) where it is on the 
    solved board
    """
    i = (val-1) // SIZE
    j = (val-1) % SIZE
    return (i,j)

def hamming(board, trash):
    """
    computes the hamming metric from the board to the solved state and returns 
    it
    trash is merely to get to conform to standard metric implementation
    https://en.wikipedia.org/wiki/Hamming_distance
    """
    dist = 0
    for i in range (0, SIZE):
        for j in range (0, SIZE):
            # check of (2 * (SIZE - 1)) ensures that do not check 
            if i + j != (2 * (SIZE - 1)) and(i,j) != get_proper_loc(board[i,j]):
                dist += 1

    return dist


def manhattan(board, trash):
    """
    computes the manhattan metric from the board to the solved stateand returns 
    it
    trash is merely to get to conform to standard metric implementation
    https://en.wikipedia.org/wiki/Taxicab_geometry
    """
    dist = 0
    for i in range (0, SIZE):
        for j in range (0, SIZE):
            (i_right, j_right) = get_proper_loc(board[i,j])
            if board[i,j] != SIZE ** 2:
                dist += abs(i - i_right) + abs(j - j_right)

    return dist
