"""
gen_train_data.py
worked on by: Luka Valencic

only run entire module as gen_train_data.py <num_processes> <output file base> 
    <number of hours to run for>
will create numprocesses files, with names of output file base plus some
    number. Runs for set number of hours before closing files and finishing
"""
import multiprocessing as multi
import numpy as np
import sys
import time as t
from solver import *
from io_help import *
from constants import *
from heuristic import *

def gen_data(run_time, file_name):
    """generates boards, finds distance to solution for each board, as well as 
    boards in solution path, and outputs to file until time is exceeded"""
    start_time = t.perf_counter()
    end_time = start_time
    file = open(file_name, "w")
    while(end_time - start_time < run_time):
        (_, _, path) = solve(gen_board(), manhattan, None)
        for i in range(0, len(path)):
            output = board_and_dist_to_string(path[i], i)
            file.write(output+'\n')
        end_time = t.perf_counter()
    file.close()

if __name__ == "__main__":
    if not (len(sys.argv) == 4):
        print("usage error: arg1 = number of processes,")
        print("arg2 = output file base,")
        print("arg3 = number of hours to run for")
        exit()
    num_processes = int(sys.argv[1])
    out_file = sys.argv[2]
    run_time = HOUR_TO_SEC * float(sys.argv[3])
    for i in range(num_processes):
        file_name = out_file + str(i) + FILE_ENDING
        p = multi.Process(target = gen_data, args = (run_time, file_name,))
        p.start()
