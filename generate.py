from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import nn_utils.network_utils as network_utils
import gen_utils.seed_generator as seed_generator
import gen_utils.sequence_generator as sequence_generator
from data_utils.parse_files import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_directory', required=True, type=str) #model file
parser.add_argument('--model', required=True, type=str) #model file
parser.add_argument('--output_file', required=True, type=str)
parser.add_argument('--hidden_dimension_size', type=int, default=1024)
parser.add_argument('--seed_len', type=int, default=1)
parser.add_argument('--max_seq_len', type=int, default=1)  #Defines how long the final song is. Total song length in samples = max_seq_len * example_len
parser.add_argument('--sampling_rate', type=int, default=44100)
args = parser.parse_args()

inputFile = args.input_directory+'/NP'
model_basename = args.input_directory+'/NPWeights'

#Load up the training data
print ('Loading training data')
#X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
#y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
#X_mean is a matrix of size (num_frequency_dims,) containing the mean for each frequency dimension
#X_var is a matrix of size (num_frequency_dims,) containing the variance for each frequency dimension
X_train = np.load(inputFile + '_x.npy')
y_train = np.load(inputFile + '_y.npy')
X_mean = np.load(inputFile + '_mean.npy')
X_var = np.load(inputFile + '_var.npy')
print ('Finished loading training data')

#Figure out how many frequencies we have in the data
freq_space_dims = X_train.shape[2]

#Creates a lstm network
model = network_utils.create_lstm_network(num_frequency_dimensions=freq_space_dims, num_hidden_dimensions=args.hidden_dimension_size)
#You could also substitute this with a RNN or GRU
#model = network_utils.create_gru_network()

#Load existing weights if available
if os.path.isfile(args.model):
	model.load_weights(args.model)
else:
	print('Model filename ' + args.model + ' could not be found!')

print ('Starting generation!')
#Here's the interesting part
#We need to create some seed sequence for the algorithm to start with
#Currently, we just grab an existing seed sequence from our training data and use that
#However, this will generally produce verbatum copies of the original songs
#In a sense, choosing good seed sequences = how you get interesting compositions
#There are many, many ways we can pick these seed sequences such as taking linear combinations of certain songs
#We could even provide a uniformly random sequence, but that is highly unlikely to produce good results
seed_seq = seed_generator.generate_copy_seed_sequence(seed_length=args.seed_len, training_data=X_train)

output = sequence_generator.generate_from_seed(model=model, seed=seed_seq, 
	sequence_length=args.max_seq_len, data_variance=X_var, data_mean=X_mean)
print ('Finished generation!')

#Save the generated sequence to a WAV file
save_generated_example(args.output_file, output, sample_frequency=args.sampling_rate)