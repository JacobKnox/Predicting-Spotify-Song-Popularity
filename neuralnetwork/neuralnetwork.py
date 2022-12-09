# neuralnetwork
"""Predict Spotify song popularity based on numerous parameters using a neural network."""

import numpy as np
import os
import pdb
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

ROOT = os.path.dirname(os.path.dirname(__file__)) # Root directory of this code

def main():
    # Get the relevant files
    datafile = os.path.expanduser(os.path.join(ROOT, "data", "data.txt"))
    labelfile = os.path.expanduser(os.path.join(ROOT, "data", "labels.txt"))
    # attributefile = os.path.expanduser(os.path.join(ROOT, "data", "attributes.txt"))

    # Load the data from these files and split them into training and testing
    x = np.loadtxt(datafile, delimiter=" ", ndmin=2)
    train_test_split = int(0.75 * len(x))
    xtrain, xtest = x[:train_test_split, :], x[train_test_split:, :]
    y = np.loadtxt(labelfile, dtype=int)
    ytrain, ytest = y[:train_test_split], y[train_test_split:]
    numSamples, numInputs = xtrain.shape
    numOutputs = len(np.unique(ytrain))

    pdb.set_trace()

    # Create the neural network
    model = Sequential()
    model.add(Input(numInputs,))
    model.add(Dense(units=500, activation='relu', name='hidden1'))
    model.add(Dense(units=500, activation='relu', name='hidden2'))
    model.add(Dense(units=numOutputs, activation='softmax', name='output'))
    model.summary()

    input("Press <Enter> to train this network...")

    model.compile(
        loss='mean_squared_error',
        optimizer=SGD(learning_rate=0.001)
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
    history = model.fit(xtrain, ytrain,
        batch_size=50,
        epochs=40,
        verbose=1,
        callbacks=[callback],
        validation_data=(xtest, ytest)
    )

if __name__ == "__main__":
    main()
    