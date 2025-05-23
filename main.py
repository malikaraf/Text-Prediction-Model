import requests
import sys
import tensorflow as tf
from dataset import preprocess_text, create_dataset
from model import build_model
from train import train
from generate import generate_text

# Load dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
encoded_text, char2idx, idx2char = preprocess_text(text)

# Hyperparameters
seq_length = 100
batch_size = 64
epochs = 2
vocab_size = len(char2idx)

# Build the model
model = build_model(vocab_size)

# Mode: 'train' or 'generate'
mode = sys.argv[1] if len(sys.argv) > 1 else "train"

if mode == "train":
    dataset = create_dataset(encoded_text, seq_length, batch_size)
    train(model, dataset, epochs)
    model.save_weights('training_checkpoints/char_model.weights.h5')

elif mode == "generate":
    # Rebuild model with batch_size=1 for generation
    model = build_model(vocab_size, batch_size=1)
    model.load_weights('training_checkpoints/char_model.weights.h5')
    model.build(tf.TensorShape([1, None]))

    # Generate text
    print(generate_text(model, start_string="ROMEO:", char2idx=char2idx, idx2char=idx2char, temperature=0.8))

else:
    print("Usage: python main.py [train|generate]")
