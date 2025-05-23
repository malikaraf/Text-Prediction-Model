import numpy as np
import tensorflow as tf

# Generates text using the trained model
def generate_text(model, start_string, char2idx, idx2char, num_generate=500, temperature=1.0):
    input_indices = [char2idx[s] for s in start_string]  # Convert input string to indices
    input_indices = tf.expand_dims(input_indices, 0)  # Add batch dimension
    text_generated = []


    for _ in range(num_generate):
        predictions = model(input_indices)  # Predict next character probabilities
        predictions = predictions[:, -1, :] / temperature  # Use temperature to scale prediction confidence
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()  # Sample from distribution

        input_indices = tf.expand_dims([predicted_id], 0)  # Update input with predicted character
        text_generated.append(idx2char[predicted_id])  # Convert index back to character

    return start_string + ''.join(text_generated)  # Return complete generated string
