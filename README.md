# 15_puzzle_learned_heuristic

# 15-puzzle representation
Puzzle is represented in a 4x4 np.array, 16 represents the blank tile, 1-15 represent the corresponding tiles

# File Contains:
-- constants.py: contains constants for different modules
-- heuristic.py: contains code for evaluating a puzzle instance based on the Manhattan and Hamming metrics
-- io_help.py: functions for converting boards and distances to strings and back
-- sovler.py: contains code for solving puzzle and generating random 15-puzzle instance
-- gen_train_data.py: module for generating training data. See module itself for how to run it