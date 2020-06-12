# 15_puzzle_learned_heuristic

# 15-puzzle representation
Puzzle is represented in a 4x4 np.array, 16 represents the blank tile, 1-15 represent the corresponding tiles

# File Contains:
- gen_train_data.py: module for generating training data. See module itself for how to run it   

# Folders
- A_Star 
  - constants.py: contains constants for different modules 
  - heuristic.py: contains code for evaluating a puzzle instance based on the Manhattan and Hamming metrics  
  - io_help.py: functions for converting boards and distances to strings and back 
  - solver.py: contains code for solving puzzle and generating random 15-puzzle instance  
- Algorithm_Models: contains saved algorithm portfolio models 
- Algorithm_Training:
  - algorithm_training.py: trains nn for algorithm portfolio 
- Data_Files: contains all generated data files 
- Model_Training: 
  - neural_net.py: contains code for training a neural net
  - xg_boost_2.py, xgboost.ipynb, xgboost_gpu.ipynb: cotains code for training an xg_boost model
- XGBoost_Models: contains saved XGBoost models 
  
