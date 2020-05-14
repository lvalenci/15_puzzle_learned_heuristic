"""
solver.py
worked on by: Luka Valencic
"""
import numpy as np
import heapq as h
import time as t
import copy

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
    return np.array_equal(board, (np.arange(16) + 1).reshape((4,4)))

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

#Deprecated, changed gen_board to not require this
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
            return (num_inv % 2 == 1)
        # if entry is in odd row counting from top, solvable if number of
        # inversions is odd
        else:
            return (num_inv % 2 == 0)

    else:
        return (num_inv % 2 == 0)
  
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

def gen_board():
    """
    generates a board in a random configuration and returns it.
    Ensures that board is solvable using board_solvable
    """
    board = (np.arange(SIZE**2)+1).reshape((SIZE, SIZE))
    for _ in range(NUM_MOVES):
        blank_pos = find_blank(board)
        moves = valid_moves(blank_pos)
        choice = np.random.choice(len(moves))
        board = make_move(board, blank_pos, moves[choice])

    return board

def solve(board, metric):
    """given a (solvable) board and a metric, this function solves the N-puzzle
    through A* search
    metric must only take in argument of board as a SIZE x SIZE np array
    board must be a SIZE x SIZE np array returned from gen_board()
    Return (n_states, tot_time, sol_path)
    n_states: the number of states visited during A* search,
        this includes the start and end states
    tot_time: the time elapsed during computation in seconds
    sol_path: a list of all the states in the shortest path to a solved 
        state, this includeds the start and end states"""
    
    # initialize all relevant variables
    # counter so elements in heap are always unique
    count = 0
    start_time = t.process_time()
    sol_path = []
    n_states = 0
    # used to store distances travelled to any solution so that do not 
    # re-enqueue if distance is worse
    visited = dict()
    queue = [(0,0,-1,[board])]
    """ element in heap as follows: 
    (estimated distance to solution, 
    current distance traveled,
    list of boards visited in this solution path - current one is at end)"""
    h.heapify(queue)

    # run actual computation
    solved = False
    while(not solved):
        (_, dist, _, path) = h.heappop(queue)
        curr_board = path[-1]

        board_rep = np.array_str(curr_board)
        # if board already visited with a lower distance, just skip it
        # if not already visited, then we are visiting a new state
        if board_rep in visited:
            if visited[board_rep] <= dist:
                continue
        else:
            visited[board_rep] = dist
            n_states += 1

        if is_solved(curr_board):
            path.reverse()
            sol_path = path
            solved = True
            break


        # generate all possible moves, execute each move, and add to queue with
        # distance metric evaluated

        blank_loc = find_blank(curr_board)
        poss_moves = valid_moves(blank_loc)

        for move in poss_moves:
            new_board = make_move(curr_board, blank_loc, move)
            est_dist = metric(new_board) + dist
            new_path = copy.deepcopy(path)
            new_path.append(new_board)
            h.heappush(queue, (est_dist, dist + 1, count, new_path))
            count += 1




    # compute compute ending time and return
    end_time = t.process_time()
    tot_time = end_time - start_time
    return (n_states, tot_time, sol_path)