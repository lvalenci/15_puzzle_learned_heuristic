# 15_puzzle_learned_heuristic

# 15-puzzle representation
Puzzle is represented in a 4x4 np.array, 16 represents the blank tile, 1-15 represent the corresponding tiles

# Folders
- A_Star 
  - constants.py: contains constants for different modules 
  - heuristic.py: contains code for evaluating a puzzle instance based on the Manhattan and Hamming metrics  
  - io_help.py: functions for converting boards and distances to strings and back 
  - solver.py: contains code for solving puzzle and generating random 15-puzzle instance  
- Algorithm_Models: contains saved algorithm portfolio models 
- Algorithm_Training
  - algorithm_training.py: trains nn for algorithm portfolio 
- Data_Files: contains all generated data files
- Data_Generation
  - comb_portfolio_data.py: combines files from gen_portfolio data to generate data for algorithm portfolio
  - gen_portfolio_data.py: outputs files of difference between actual distance to solution and predicted distance for use in algorithm training
  - gen_test_boards.py: generates datafile of boards, distances to solution, and states explored when finding solution
  - gen_train_data.py: generates dataset of randome boards and distances to solution
- Model_Summaries
  - model_performances.md: contains information about performance of nn models on training set
  - model_summaries.md: contains information about in-sample performances and architectures of nn models
- Model_Training
  - neural_net.py: contains code for training a neural net
  - xg_boost_2.py, xgboost.ipynb, xgboost_gpu.ipynb: contains code for training an xg_boost model
- KNN Models
  - knn_find_best.ipynb: contains code for training and testing KNN Classifiers with various parameters.
  - knn_final.ipynb: contains code for training and testing best performing KNN Classifier.
  - board_rep_4.ipynb: contains code for training and testing neural net with features generated using final KNN.
- Testing
  - run_algorithm_portfolio.py: runs testing for an algorithm portfolio
  - run_testing_nn.py: runs testing on a neural net model
  - run_testing_xg.py: runs testing on a xgboost model
- XGBoost_Models: contains saved XGBoost models 
  
