#!/usr/bin/python3

import tensorflow as tf
from tensorflow import keras

from datetime import datetime
import numpy as np

import os

os.environ["HIP_VISIBLE_DEVICES"]="0"
# os.environ["HCC_PROFILE"]="2"


print("TensorFlow version: ", tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# data normalization
train_images = train_images / 255
test_images = test_images / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Take a look at the model summary
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

log_dir="logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 3)

model.fit(train_images, train_labels, epochs=10, callbacks=[tensorboard_callback])

# Evaluate the model on test set
score = model.evaluate(test_images, test_labels, verbose=2)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])
