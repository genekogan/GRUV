from data_utils.parse_files import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_directory', required=True, type=str)
parser.add_argument('--sampling_rate', type=int, default=44100)
parser.add_argument('--clip_len', type=int, default=10)
args = parser.parse_args()

output_filename = args.input_directory+'/NP'
block_size = args.sampling_rate / 4  #block sizes used for training - this defines the size of our input state
max_seq_len = int(round((args.sampling_rate * args.clip_len) / block_size)) #Used later for zero-padding song sequences

#Step 1 - convert MP3s to WAVs
new_directory = convert_folder_to_wav(args.input_directory+'/', args.sampling_rate)

#Step 2 - convert WAVs to frequency domain with mean 0 and standard deviation of 1
convert_wav_files_to_nptensor(new_directory, block_size, max_seq_len, output_filename)