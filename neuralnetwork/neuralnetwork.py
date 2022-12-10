# neuralnetwork
"""Predict Spotify song popularity based on numerous parameters using a neural network."""

import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

ROOT = os.path.dirname(os.path.dirname(__file__)) # Root directory of this code

def main():
    # Get the relevant files
    datafile = os.path.expanduser(os.path.join(ROOT, "data", "data.txt"))
    labelfile = os.path.expanduser(os.path.join(ROOT, "data", "labels.txt"))
    # attributefile = os.path.expanduser(os.path.join(ROOT, "data", "attributes.txt"))

    # Load the data from these files and split them into training and testing
    x = np.loadtxt(datafile, delimiter=" ", ndmin=2)
    train_test_split = int(0.75 * len(x))
    x_train, x_test = x[:train_test_split, :], x[train_test_split:, :]
    y = np.loadtxt(labelfile, dtype=int)
    y_train, y_test = y[:train_test_split], y[train_test_split:]
    numSamples, numInputs = x_train.shape
    maxOutput = np.max(y_train) + 1
    t_train = to_categorical(y_train, maxOutput)  # convert output to categorical targets
    t_test = to_categorical(y_test, maxOutput)

    pdb.set_trace()

    # Create the neural network
    model = Sequential()
    model.add(Input(numInputs,))
    model.add(Dense(units=100, activation='relu', name='hidden1'))
    model.add(Dense(units=200, activation='relu', name='hidden2'))
    model.add(Dense(units=100, activation='relu', name='hidden3'))
    model.add(Dense(units=maxOutput, activation='softmax', name='output'))
    model.summary()

    input("Press <Enter> to train this network...")

    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Add an Early Stopping callback
    callback = EarlyStopping(
        monitor='loss',
        min_delta=1e-4,
        patience=5,
        verbose=1
    )

    # Train the network
    history = model.fit(x_train, t_train,
        batch_size=10,
        epochs=100,
        verbose=1,
        callbacks=[callback],
        validation_data=(x_test, t_test)
    )

    # Compute the accuracy
    metrics_train = model.evaluate(x_train, t_train, verbose=0)
    print("=================================")
    print(f"Training loss = {metrics_train[0]:0.4f}")
    print(f"Training accuracy = {metrics_train[1]:0.4f}")
    
    metrics_test = model.evaluate(x_test, t_test, verbose=0)
    print(f"Testing loss = {metrics_test[0]:0.4f}")
    print(f"Testing accuracy = {metrics_test[1]:0.4f}")

    # Plot the performance over time
    plt.subplots()
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='testing accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()
    