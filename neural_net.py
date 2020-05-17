import numpy as np
import sys
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
from constants import * 
from heuristic import *
from io_help import *
from solver import *

def neural_net_heuristic(board, model):

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

def train(file_name):
	"""
	This function reads in training data from a file and returns a 
	trained NN model. 
	"""
	file = open(file_name, "r")

	train_X = []
	train_Y = []

	for string in file: 
		(board, dist) = string_to_board_and_dist(string) 

		train_X.append(one_hot_encode(board))
		train_Y.append(dist)

	file.close()

	# Build Model
	model = Sequential()

	# Input Layer
	model.add(Dense(units=100, input_dim=256, activation='relu'))
	model.add(Dropout(0.1))
	# Hidden Layers
	model.add(Dense(units=100, activation='relu'))
	# Output Layer
	model.add(Dense(units=1, activation='linear'))

	# Define the optimizer and loss function
	model.compile(optimizer='adam', loss='mse')

	# You can also define a custom loss function
	# model.compile(optimizer='adam', loss=custom_loss)

	# Train 
	model.fit(np.array(train_X), np.array(train_Y), epochs=100)

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
