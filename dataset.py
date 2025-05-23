import numpy as np
import tensorflow as tf

# Converts text into a list of unique characters and encodes each character into an integer
def preprocess_text(text):
    chars = sorted(set(text))  # Sorted list of unique characters
    char2idx = {u: i for i, u in enumerate(chars)}  # Mapping from char to index
    idx2char = np.array(chars)  # Mapping from index to char
    encoded_text = np.array([char2idx[c] for c in text])  # Encode entire text as integers
    return encoded_text, char2idx, idx2char

# Creates training dataset by forming sequences of fixed length from encoded text
def create_dataset(encoded_text, seq_length, batch_size):
    char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)  # Turn array into dataset of chars
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)  # Make sequences of length seq_length + 1

    # Function to split each sequence into input and target
    def split_input_target(chunk):
        input_text = chunk[:-1]  # Input is all chars except the last
        target_text = chunk[1:]  # Target is all chars except the first
        return input_text, target_text
  
    dataset = sequences.map(split_input_target)  # Map split function to all sequences
    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)  # Shuffle and batch the dataset
    return dataset
