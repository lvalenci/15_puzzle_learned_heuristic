"""
make_test_boards.py
worked on by: Luka Valencic
"""
from constants import *
from solver import gen_board
from io_help import board_to_string

if __name__ == "__main__":
    file = open("Test_boards.txt", "w")

    for _ in range(NUM_TEST_BOARDS):
        board = gen_board()
        file.write(board_to_string(board) + '\n')
    file.close()