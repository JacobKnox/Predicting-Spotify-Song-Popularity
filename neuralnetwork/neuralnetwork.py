# neuralnetwork
"""Predict Spotify song popularity based on numerous parameters using a neural network."""

import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import SGD, Adam

ROOT = os.path.dirname(os.path.dirname(__file__)) # Root directory of this code

def main():
    # Get the relevant files
    datafile = os.path.expanduser(os.path.join(ROOT, "data", "data.txt"))
    labelfile = os.path.expanduser(os.path.join(ROOT, "data", "labels.txt"))

    # Load the data from these files
    x = np.loadtxt(datafile, delimiter=" ", ndmin=2)
    y = np.loadtxt(labelfile, dtype=int)
    
    # Delete some data that doesn't seem necessary 
    x = np.delete(x, 2, 1)
    x = np.delete(x, 5, 1)
    
    # Scale data
    _, inputs = x.shape
    for i in range(inputs):
        x[:, i] = x[:, i] / np.max(x[:, i])
    y_scaled = y / np.max(y) # Scale the target labels to be between 0-1
    
    # split data and labels into training and testing
    train_test_split = int(0.75 * len(x))
    x_train, x_test = x[:train_test_split, :], x[train_test_split:, :]
    y_train, y_test = y_scaled[:train_test_split], y_scaled[train_test_split:]
    _, numInputs = x_train.shape

    # Create the neural network
    model = Sequential()
    model.add(Input(shape=(numInputs,)))
    model.add(Dense(units=500, activation='relu', name='hidden1'))
    model.add(Dense(units=500, activation='relu', name='hidden2'))
    model.add(Dense(units=500, activation='sigmoid', name='hidden3'))
    model.add(Dense(units=1, activation='sigmoid', name='output'))
    model.summary()

    input("Press <Enter> to train this network...")

    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.02)
    )

    # Train the network
    history = model.fit(x_train, y_train,
        batch_size=50,
        epochs=100,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    # Compute the loss
    metrics_train = model.evaluate(x_train, y_train, verbose=0)
    print("=================================")
    print(f"Training loss = {metrics_train:0.4f}")
    
    metrics_test = model.evaluate(x_test, y_test, verbose=0)
    print(f"Testing loss = {metrics_test:0.4f}")

    # Plot the performance over time
    plt.subplots()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='testing loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.show(block=False)

    pred = model.predict(x_test)
    # differences = np.absolute(pred.reshape(len(pred),) - y_test)
    print(pred)

if __name__ == "__main__":
    main()
    