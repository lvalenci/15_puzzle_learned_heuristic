"""
xg_boost.py
worked on by: Yasmin Veys

Run xg_boost.py <training_file_input> 
EX: python3 xg_boost.py Meena_5_16_89475.txt saved_model.txt
"""
import numpy as np
import sys
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from constants import * 
from heuristic import *
from io_help import *
from solver import *

def xgboost_heuristic(board, model):

	"""
	This function takes in a board and a trained NN model and returns
	the heuristic the model predicts.
	"""

	return model.predict(one_hot_encode(board).reshape(1,256))

def one_hot_encode(board):
	""" 
	This function one hot encodes the board into a length 256 array.
	The one hot encoding gives the location of each number in the board.
	For example, the first 16 of the 256 numbers will indicate where on
	the board the 1 tile is. 
	"""

	flat = (board.reshape(SIZE ** 2)).tolist()
	
	X = []
	for i in np.arange(1,17): 
		encoding = np.zeros(SIZE ** 2)
		encoding[flat.index(i)] = 1

		X.append(encoding)	

	X = (np.asarray(X).reshape(SIZE ** 4))

	# Potentially append Manhattan distance. 
	# np.append(X, manhattan(board))

	return X

def load_data(file_name):
	"""
	This function reads in training data from a file and returns 
	the one-hot encoded data X and their labels Y as a tuple. 
	"""
	file = open(file_name, "r")

	X = []
	Y = []

	for string in file: 
		(board, dist) = string_to_board_and_dist(string) 

		X.append(one_hot_encode(board))
		Y.append(dist)

	file.close()

	return(np.asarray(X),np.asarray(Y))

def evaluate(file_name):
	"""
	This function reads in training data from a file and 
	trains and evaluates an xgboost model using kfold validation. 
	"""
	(X,Y) = load_data(file_name)

	# Implement K-fold cross validation
	kfold = KFold(n_splits=10, shuffle=True, random_state=2020)

	# Build Model, Look at the link below for parameter options
	# https://xgboost.readthedocs.io/en/latest/python/python_api.html
	model = XGBClassifier(learning_rate=0.08, max_depth=3, min_child_weight=2, 
		n_estimators=110, nthread=4, objective='reg:logistic', subsample=1, verbosity=0)

	score = cross_val_score(model, X, Y, cv=kfold)
	print(score.mean()*100)

	return model

def train(file_name):
	"""
	This function reads in training data from a file and returns a 
	trained xgboost model. 
	"""
	(X,Y) = load_data(file_name)

	# Build Model, Look at the link below for parameter options
	# https://xgboost.readthedocs.io/en/latest/python/python_api.html
	model = XGBClassifier(learning_rate=0.08, max_depth=3, min_child_weight=2, 
		n_estimators=110, nthread=4, objective='reg:logistic', subsample=1, verbosity=2)

	# Train 
	model.fit(X, Y)

	return model

def run_saved_model(model_file, data_file):
	"""
	given a file to which a model is saved a a datafile, runs model on data
	on code to evaluate accuracy and score
	"""
	model = pickle.load(open(model_file, "rb"))
	(X, Y) = load_data(data_file)
	model._le = LabelEncoder().fit(Y)
	y_pred = model.predict(X)
	pred = [round(v) for v in y_pred]
	acc = accuracy_score(Y, pred)
	print(acc * 100.0)

if __name__ == "__main__":

	if not (len(sys.argv) == 3):
		print("usage error: \narg1 = input file name\narg2 = save model file")
		exit()

	file_name = sys.argv[1] 
	save_file = sys.argv[2]

	# Toy Example Testing 
	# To train on the entire data set, replace evaluate with train
	model = train(file_name)
	board = gen_board()

	pickle.dump(model, open(save_file, "wb"))

	run_saved_model(save_file, "All_Data.txt")
	#print(xgboost_heuristic(board, model))
	#print(manhattan(board))
