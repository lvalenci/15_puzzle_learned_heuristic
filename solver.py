"""
solver.py
worked on by: Luka Valencic
"""
import numpy as np
import heapq as h
from constants import *

def find_blank(board):
    """
    given a board, returns the location (i,j) of the blank tile
    """
    blank_val = SIZE ** 2
    for i in range(0, SIZE):
        for j in range(0, SIZE):
            if board[i,j] == blank_val:
                return (i,j)

def is_solved(board):
    """
    given a board, returns true if the board is solved, false otherwise
    """
    return all(board == (np.arange(SIZE ** 2) + 1))

def inversion_count(board):
    """
    note this takens in a flattened board
    finds the number of inversions (x before y  when x > y)
    """
    num_elem = SIZE ** 2
    num_inv = 0
    for i in range(0, num_elem - 1):
        for j in range(i + 1, num_elem):
            if (board[i] != num_elem and board[j] != num_elem and board[i] > board[j]):
                num_inv += 1

    return num_inv

def board_solvable(board):
    """
    returns true if a board is solvable, false otherwise
    """
    (i, _) = find_blank(board)
    num_inv = inversion_count(board.reshape(SIZE ** 2))

    if (SIZE % 2 == 0):
        # if entry in even row counting from top, solvable if number of
        # inversions is odd
        if (i % 2 == 0):
            return (num_inv % 2 == 0)
        # if entry is in odd row counting from top, solvable if number of
        # inversions is odd
        else:
            return (num_inv % 2 == 1)

    else:
        return (num_inv % 2 == 0)
  
def rand_board():
    """
    generates a random board, does not check for solvability
    """  
    board = np.random.permutation(SIZE ** 2) + 1
    board = board.reshape((SIZE, SIZE))
    return board

def gen_board():
    """
    generates a board in a random configuration and returns it.
    Ensures that board is solvable using board_solvable
    """
    made_solvable = False
    board = None
    while(not made_solvable):
        board = rand_board()
        made_solvable = board_solvable(board)
    return board

def make_move(board, old_cord, new_cord):
    """
    given a board and two indices on the board, swaps the values at the two 
    indicies and returns the board. 
    The original board is unchanged and a copy is returned
    Assumes the indices are valid configurations
    """
    board = board.copy()
    (oi, oj) = old_cord
    (ni, nj) = new_cord
    temp = board[oi, oj]
    board[oi, oj] = board[ni, nj]
    board[ni, nj] = temp
    return board

def valid_moves(blank_pos):
    """
    give a board and the position of the blank tile, finds and returns a list
    of all valid moves, each move is represented by a tuple (i,j) representing
    a location on the board to which the blank square can move to"""
    (i, j) = blank_pos
    moves = []
    if(i > 0):
        moves.append((i-1, j))
    if(i < SIZE-1):
        moves.append((i+1, j))
    if(j > 0):
        moves.append((i, j-1))
    if(j < SIZE-1):
        moves.append((i, j+1))

    return moves