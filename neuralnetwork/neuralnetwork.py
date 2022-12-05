# neuralnetwork
"""Predict Spotify song popularity based on numerous parameters using a neural network."""

import numpy as np
import os
import pdb

ROOT = os.path.dirname(os.path.dirname(__file__)) # Root directory of this code

def main():
    # Get the relevant files
    datafile = os.path.expanduser(os.path.join(ROOT, "data", "data.txt"))
    labelfile = os.path.expanduser(os.path.join(ROOT, "data", "labels.txt"))
    # attributefile = os.path.expanduser(os.path.join(ROOT, "data", "attributes.txt"))

    # Load the data from these files
    x = np.loadtxt(datafile, delimiter=" ", ndmin=2)
    t = np.loadtxt(labelfile, dtype=int)

    pdb.set_trace()

if __name__ == "__main__":
    main()
    