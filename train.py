import tensorflow as tf
import os

# Trains the model and saves checkpoints
def train(model, dataset, epochs, checkpoint_dir='training_checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='training_checkpoints/char_model.weights.h5',
    save_weights_only=True,
    save_best_only=False
)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')  # Compile model with loss and optimizer
    model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])  # Train model over multiple epochs

