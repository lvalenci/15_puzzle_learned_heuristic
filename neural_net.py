"""
neural_net.py
worked on by: Yasmin Veys

Run neural_net.py <training_file_input> 
EX: python3 neural_net.py Meena_5_16_89475.txt
"""

import numpy as np
import sys
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
from sklearn.model_selection import KFold

from constants import * 
from heuristic import *
from io_help import *
from solver import *

def neural_net_heuristic(board, model):

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

def evaluate(file_name):
	"""
	This function reads in training data from a file and returns a 
	trained NN model. 
	"""
	np.random.seed(2020)

	file = open(file_name, "r")

	X = []
	Y = []

	for string in file: 
		(board, dist) = string_to_board_and_dist(string) 

		X.append(one_hot_encode(board))
		Y.append(dist)

	file.close()

	X = np.asarray(X)
	Y = np.asarray(Y)

	# Implement K-fold cross validation
	kfold = KFold(n_splits=10, shuffle=True, random_state=2020)

	for train, test in kfold.split(X, Y):
		# Build Model
		model = Sequential()

		# Input Layer
		model.add(Dense(units=256, input_dim=256, activation='relu'))
		model.add(Dropout(0.1))
		# Hidden Layers
		model.add(Dense(units=256, activation='relu'))
		# Output Layer
		model.add(Dense(units=1, activation='linear'))

		# Define the optimizer and loss function
		model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

		# You can also define a custom loss function
		# model.compile(optimizer='adam', loss=custom_loss)

		# Train 
		model.fit(X[train], Y[train], epochs=20)

		# Evaluate
		score = model.evaluate(X[test], Y[test], verbose=0)
		print(score)

	return model

def train(file_name):
	"""
	This function reads in training data from a file and returns a 
	trained NN model. 
	"""
	np.random.seed(2020)

	file = open(file_name, "r")

	X = []
	Y = []

	for string in file: 
		(board, dist) = string_to_board_and_dist(string) 

		X.append(one_hot_encode(board))
		Y.append(dist)

	file.close()

	X = np.asarray(X)
	Y = np.asarray(Y)

	# Build Model
	model = Sequential()

	# Input Layer
	model.add(Dense(units=256, input_dim=256, activation='relu'))
	model.add(Dropout(0.1))
	# Hidden Layers
	model.add(Dense(units=256, activation='relu'))
	# Output Layer
	model.add(Dense(units=1, activation='linear'))

	# Define the optimizer and loss function
	model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

	# You can also define a custom loss function
	# model.compile(optimizer='adam', loss=custom_loss)

	# Train 
	model.fit(X, Y, epochs=20)

	return model


def custom_loss(y_true, y_pred): 
	"""
	This custom loss function takes in y_true and y_pred and returns
	the loss. It cannot take more than two arguments.
	"""

	# Slighly different verson of mse, for example 
	loss = K.square((y_pred - y_true)/10)
	loss = K.mean(loss, axis=1)

	return loss

if __name__ == "__main__":

	if not (len(sys.argv) == 2):
		print("usage error: arg1 = input file name")
		exit()

	file_name = sys.argv[1] 

	# Toy Example Testing 
	model = train(file_name)
	board = gen_board()
	print(neural_net_heuristic(board, model))
	print(manhattan(board))
