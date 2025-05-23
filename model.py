import tensorflow as tf
from keras.models  import Sequential
from keras.layers  import LSTM, Dense, Embedding
  
# Defines the RNN model architecture
def build_model(vocab_size, embedding_dim=256, rnn_units=1024, batch_size=None):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(batch_shape=[batch_size, None]),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True,
            stateful=(batch_size is not None),
            recurrent_initializer='glorot_uniform'
        ),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model