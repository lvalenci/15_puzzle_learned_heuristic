"""
make_test_boards.py
worked on by: Luka Valencic
"""
from constants import *
import heuristic as h
import io_help as io
import solver as s

if __name__ == "__main__":
    file = open("Test_boards.txt", "w")

    for i in range(NUM_TEST_BOARDS):
        if (i % 50 == 0):
            print(i)
        board = s.gen_board()
        (c_states, c_time, c_path) = s.solve(board, h.manhattan, None)
        info_rep = io.test_info_to_string(board, c_states, c_time, len(c_path) - 1)
        file.write(info_rep + "\n")
        
    file.close()