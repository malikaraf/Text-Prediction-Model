# Text-Prediction-Model
Character-level Text prediction Model 

## 📁 Project Structure
 main.py # Entry point: preprocesses text, trains model, and generates sample output
├── dataset.py # Functions for encoding and batching the text data
├── model.py # Builds the LSTM-based text generation model
├── train.py # Trains the model and saves checkpoints
├── generate.py # Generates text using a trained model
├── requirements.txt # Python dependencies
├── training_checkpoints/ # Directory to store model weights
└── README.md # This file

Model uses Embedding + LSTM + Dense layers
Temperature controls randomness in generation (0.5 = conservative, 1.0 = creative)
Make sure to save weights with .h5 extension for Keras 3 compatibility

## Dependencies 

- Tensorflow
- Numpy
- OS
- Requests( i am new to this so maybe requests and os isn't really needed to be mentioned in the document)

PS - Version mismatches are a big thing. Check if the tensorflow version goes with the python version you have installed. Latest python versions might not necessarily support tensorflow. Check the model of keras as well since you do need to work with Sequencial, LSTM,Embedding and Dense in the model.py file inorder to set up the architecture 



However, i was unable to upload the training_checkpoints folder due to storage issues. I have thus uploaded the picture instead. 

##  Getting Started

### 1. Clone the repository

- git clone and you know the rest

### 2. Set up Virtual Environment


  python -m venv .venv
# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate


### 3. Install dependencies 
pip install -r requirements.txt

### Train the model
command: python main.py
This will execute the following cycle:

Download Shakespeare’s text

Preprocess and encode the data

Train the model over 10 epochs

Save model weights to training_checkpoints/

Generate text starting with "ROMEO:"

If the model has already been trained and weights exist, it will skip training and use the saved weights

## Generate Text Only
To skip training and just generate text:

# In main.py
# Comment out: train(model, dataset, epochs)
# Ensure this line runs only if weights are available:
model.load_weights('training_checkpoints/char_model.weights.h5')
print(generate_text(model, start_string="ROMEO:", char2idx=char2idx, idx2char=idx2char, temperature=0.




-
