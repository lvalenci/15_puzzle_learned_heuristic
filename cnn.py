"""
cnn.py
worked on by: Tarini Singh; Yasmin Veys and Luka Valencic (base code)
base code: taken from neural_net.py

Run neural_net.py <training_file_input> 
EX: python3 neural_net.py Meena_5_16_89475.txt saved_model.txt
"""

import numpy as np
import sys
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.models import load_model
from keras.utils import to_categorical
import keras.losses

from constants import * 
from heuristic import *
from io_help import *
from solver import *

# Must define all custom loss function here
def custom_loss(y_true, y_pred): 
	"""
	This custom loss function takes in y_true and y_pred and returns
	the loss. It cannot take more than two arguments.
	"""

	# Slighly different verson of mse, for example 
	loss = K.square((y_pred - y_true)/10)
	loss = K.mean(loss, axis=1)

	return loss

def exp_loss(y_true, y_pred):
	"""
	Custom loss function. 
	"""
	loss = K.exp((y_pred - y_true))
	loss = loss + K.square((y_pred - y_true) / 2)
	loss = K.mean(loss, axis = 1)

	return loss

# define custom loss functions in keras so can load model
keras.losses.custom_loss = custom_loss
keras.losses.exp_loss = exp_loss

def cnn_heuristic(board, model):
	"""
	This function takes in a board and a trained CNN model and returns
	the heuristic the model predicts.
	"""
	 # transform board into correct shape
	if is_solved(board):
		return 0
	board_size = len(board)
	X = []
	X.append(np.asarray(board))
	X = np.asarray(X)
	# divide by 16 to get "greyscale" values
	X = X.reshape(1, board_size, board_size, 1) / 16

	[probs] = model.predict(X)

	pred = np.argmax(probs) + 1
    
	return pred

def find_over_estimate(file_name, model_file):
	"""
	This function takes in a model saved in model_file and data points in 
	file_name and prints out the percentage of times said model predicted 
	a distance greater than the actual distance and the percentage of times
	said model predicted a distance less than the Manhattan Distance
	"""
	model = load_model(model_file)
	data = open(file_name, "r")
	over = []
	under = []

	for line in data:
		(board, dist) = string_to_board_and_dist(line)
		man_dist = manhattan(board)
		pred = cnn_heuristic(board, model)
		print(pred)
		over.append(pred > dist)
		under.append(pred < man_dist)

	print("prediction less than manhattan percent of the time", sum(under) * 100 / len(under))
	print("prediction greater than actual distance precent of the time", sum(over) * 100 / len(over))

def load_data(file_name):
	"""
	This function reads in training data from a file and returns 
	the boards in X and their labels in Y as a tuple. 
	"""
	file = open(file_name, "r")

	X = []
	Y = []

	for string in file: 
		(board, dist) = string_to_board_and_dist(string) 
		X.append(np.asarray(board))
		Y.append(dist)
    

	file.close()
    
	# now, need to transform the data into the correct shape
	board_size = len(X[0]) # assume training data has at least one board
	X_train = np.asarray(X)
	# divide by 16 to get "greyscale" values
	X_train = X_train.reshape(len(X), board_size, board_size, 1) / 16
    
	# need to one-hot encode our output
	# subtract 1 because to_categorical is zero-indexed
	Y_train = np.asarray(Y) - 1 
	 # num_classes is max number of moves for a board of this size
	Y_train = to_categorical(Y_train, num_classes = 80)
    
	return(X_train, Y_train)

def evaluate(file_name):
	"""
	This function reads in training data from a file and 
	trains and evaluates a CNN model using kfold validation. 
	"""
	(X,Y) = load_data(file_name)

	# Implement K-fold cross validation
	kfold = KFold(n_splits=10, shuffle=True, random_state=2020)

	# CNN features
	filters = 20
	kernel_size = (3, 3) # looks at all neighbors
	input_shape = (4, 4, 1) # last number will always be 1 bc greyscale
	padding = "valid" # "valid" -> no padding; "same" -> padding
    
	# feature of the board
	max_moves = 80
    
	for train, test in kfold.split(X, Y):
		# Build Model
		model = Sequential()

		# Input Layer
		model.add(Conv2D(
			filters=filters, 
			kernel_size=kernel_size, 
			padding=padding,
			activation='relu', 
			input_shape=input_shape))
        
		# Hidden Layers
		model.add(Flatten())
		model.add(Dense(units=256))
		model.add(Dropout(0.1))
		model.add(Dense(units=256, activation='relu'))
        
		# Output Layer
		model.add(Dense(units=max_moves, activation='softmax'))

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
	trained CNN model. 
	"""
	(X,Y) = load_data(file_name)
    
	# CNN features
	filters = 20
	kernel_size = (3, 3) # looks at all neighbors
	input_shape = (4, 4, 1) # last number will always be 1 bc greyscale
	padding = "valid" # "valid" -> no padding; "same" -> padding

	# feature of the board
	max_moves = 80
    
	# Build Model
	model = Sequential()
    
	# Input Layer
	model.add(Conv2D(
		filters=filters, 
		kernel_size=kernel_size, 
		padding=padding,
		activation='relu', 
		input_shape=input_shape))
        
	# Hidden Layers
	model.add(Flatten())
	model.add(Dropout(0.1))
	model.add(Dense(units=256, activation='relu'))
    
	# Output Layer
	model.add(Dense(units=max_moves, activation='softmax'))

	# Define the optimizer and loss function
	model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

	# You can also define a custom loss function
	# model.compile(optimizer='adam', loss=custom_loss)

	# Train 
	model.fit(X, Y, epochs=20)

	return model

def train_custom_loss(file_name, loss_func):
	"""
	This function reads in training data from a file and returns a 
	trained CNN model. This CNN model is trained with specificed loss
	function
	"""
	(X,Y) = load_data(file_name)

	# CNN features
	filters = 20
	kernel_size = (3, 3) # looks at all neighbors
	input_shape = (4, 4, 1) # last number will always be 1 bc greyscale
	padding = "valid" # "valid" -> no padding; "same" -> padding

	# feature of the board
	max_moves = 80
    
	# Build Model
	model = Sequential()
    
	# Input Layer
	model.add(Conv2D(
		filters=filters, 
		kernel_size=kernel_size, 
		padding=padding,
		activation='relu', 
		input_shape=input_shape))
        
	# Hidden Layers
	model.add(Flatten())
	model.add(Dropout(0.1))
	model.add(Dense(units=256, activation='relu'))
    
	# Output Layer
	model.add(Dense(units=max_moves, activation='softmax'))

	# Define the optimizer and loss function
	model.compile(optimizer='adam', loss=exp_loss, metrics=['accuracy'])

	# You can also define a custom loss function
	# model.compile(optimizer='adam', loss=custom_loss)

	# Train 
	model.fit(X, Y, epochs=20)

	return model

def run_saved_model(model_file, data_file):
	"""
	given a file to which a model is saved a a datafile, runs model on data
	on code to evaluate accuracy and score
	"""
	model = load_model(model_file)
	(X, Y) = load_data(data_file)
	score = model.evaluate(X, Y, verbose = 0)
	print(score)

if __name__ == "__main__":

	if not (len(sys.argv) == 3):
		print("usage error:\n arg1 = input file name \n arg2 = file to save model to")
		exit()

	file_name = sys.argv[1] 
	out_file = sys.argv[2]

	# Toy Example Testing 
	# To train on the entire data set, replace evaluate with train
	model = evaluate(file_name)
	model.save(out_file)
	board = gen_board()
	print(cnn_heuristic(board, model))
	print(manhattan(board))