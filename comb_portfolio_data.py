"""
comb_portfolio_data.py
worked on by: Luka
combines various files output by gen_portfolio_data into single file for
training algorithm portfolio
"""
import numpy as np
import io_help as io

num_points = 395715

files = ["portfolio_data_shift_mse_nn_type_2_Luka.txt", "portfolio_data_shift_mse_nn_type_2_complex_Luka.txt", \
            "portfolio_data_shift_mse_nn_type_3_complex_Luka.txt", "portfolio_data_xgboost_500_clean.txt", "portfolio_data_manhattan.txt"]
# first model in nn_shift_mse2 second is in nn_shift_mse3, don't know about 3 and 4, five is simple manhattan distance

if __name__ == "__main__":
    best_model = np.zeros(num_points) + 4
    curr_mins = np.zeros(num_points) + 10000

    model = 0
    for file in files:
        i = 0
        c_file = open(file, 'r')
        for line in c_file:
            (_, val) = io.string_to_board_and_dist(line)
            if val < curr_mins[i] and val >= 0:
                best_model[i] = model
                curr_mins[i] = val
            i += 1
        model += 1
        c_file.close()

    out = open("final_portfolio_data.txt", 'w')
    in_file = open(files[0], 'r')
    i = 0
    for line in in_file:
        (board, _) = io.string_to_board_and_dist(line)
        to_write = io.board_and_dist_to_string(board, int(best_model[i]))
        out.write(to_write + '\n')
        i += 1
    out.close()
    in_file.close()