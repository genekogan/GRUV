from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import nn_utils.network_utils as network_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_directory', required=True, type=str)
parser.add_argument('--hidden_dimension_size', type=int, default=1024)
parser.add_argument('--num_iters', type=int, default=5000) #Number of iterations for training
parser.add_argument('--epochs_per_iter', type=int, default=500) #Number of iterations before we save our model
parser.add_argument('--batch_size', type=int, default=100)	#Number of training examples pushed to the GPU per batch. Larger batch sizes require more memory, but training will be faster
args = parser.parse_args()

inputFile = args.input_directory+'/NP'
cur_iter = 0
model_basename = args.input_directory+'/NPWeights'
model_filename = model_basename + str(cur_iter)

#Load up the training data
print ('Loading training data')
#X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
#y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
X_train = np.load(inputFile + '_x.npy')
y_train = np.load(inputFile + '_y.npy')
print ('Finished loading training data')

#Figure out how many frequencies we have in the data
freq_space_dims = X_train.shape[2]

#Creates a lstm network
model = network_utils.create_lstm_network(num_frequency_dimensions=freq_space_dims, num_hidden_dimensions=args.hidden_dimension_size)
#You could also substitute this with a RNN or GRU
#model = network_utils.create_gru_network()

#Load existing weights if available
if os.path.isfile(model_filename):
	model.load_weights(model_filename)

print ('Starting training!')
while cur_iter < args.num_iters:
	print('Iteration: ' + str(cur_iter))
	#We set cross-validation to 0,
	#as cross-validation will be on different datasets 
	#if we reload our model between runs
	#The moral way to handle this is to manually split 
	#your data into two sets and run cross-validation after 
	#you've trained the model for some number of epochs
	history = model.fit(X_train, y_train, batch_size=args.batch_size, nb_epoch=args.epochs_per_iter, verbose=1, validation_split=0.0)
	cur_iter += args.epochs_per_iter
	model.save_weights(model_basename + '_'+str(cur_iter)+'iter')

print ('Training complete!')
