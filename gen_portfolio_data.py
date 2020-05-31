"""
gen_portfolio_data.py
worked on by: Luka Valencic
Generates training data for algorithm portfolio
Output data is in the form of a standard board rep and (true distance - predicted distance)
Modify main where specified to get it to work with your model
"""
import sys
import numpy as np

import io_help as io
import heuristic as h
import neural_net as nn


def gen_portfolio_data(out_file, model, h_func):
    """for all datapoints in All_Data.txt, writes new file containing difference
    between actual distance and distance predicted by model"""
    output = open(out_file, "w")
    data = open("All_Data.txt", "r")

    for line in data:
        (board, dist) = io.string_to_board_and_dist(line)
        pred = h_func(board, model)
        diff = dist - pred
        n_line = io.board_and_dist_to_string(board, diff)
        output.write(n_line + '\n')
    data.close()
    output.close()

if __name__ == "__main__":
    if not (len(sys.argv) == 2):
        print("usage error:\n arg1 = file to which data should be output")

    # modify section so that h_func and model are correct for your instance
    h_func = h.manhattan
    model = None
    # end of section to be modified
    gen_portfolio_data(sys.argv[1], model, h_func)