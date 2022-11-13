# decisiontree.py
"""Predict Parkinson's disease based on dysphonia measurements using a decision tree."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix

path = os.getcwd()
ROOT = os.path.dirname(path)  # root directory of this code


def main():
    # Relevant files
    datafile = os.path.expanduser(os.path.join(ROOT, "data", "data.txt"))
    labelfile = os.path.expanduser(os.path.join(ROOT, "data", "labels.txt"))
    attributesfile = os.path.expanduser(os.path.join(ROOT, "data", "attributes.txt"))

    # Load data from relevant files
    xtrain = np.loadtxt(datafile, dtype = float, delimiter=",", ndmin=2)
    ytrain = np.loadtxt(labelfile, dtype = int)
    attributes = np.loadtxt(attributesfile, dtype = str)

    # Train a decision tree via information gain on the training data
    clf = DecisionTreeClassifier()
    clf = clf.fit(xtrain, ytrain)

    # Test the decision tree
    train = clf.predict(xtrain)

    # Compare training and test accuracy
    trainerror = np.abs(ytrain - train)
    print("Training Accuracy: " + str((len(trainerror) - sum(trainerror)) / len(trainerror)))

    # Visualize the tree using matplotlib and plot_tree
    fig = plt.figure(figsize=(50,50))
    plot_tree(clf, feature_names = attributes, filled = True, fontsize = 28, class_names = ["0", "1"], rounded = True)
    fig.show()
    fig.savefig("decistion_tree.png")


if __name__ == '__main__':
    main()
