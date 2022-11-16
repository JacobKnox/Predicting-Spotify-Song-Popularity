# decisiontree.py
"""Predict Spotify song popularity based on numerous parameters using a decision tree."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from sklearn.tree import plot_tree
from utils import get_best_tree

path = os.getcwd()
ROOT = os.path.dirname(path)  # root directory of this code


def main():
    # Relevant files
    datafile = os.path.expanduser(os.path.join(ROOT, "data", "data.txt"))
    labelfile = os.path.expanduser(os.path.join(ROOT, "data", "labels.txt"))
    attributesfile = os.path.expanduser(os.path.join(ROOT, "data", "attributes.txt"))

    # Load data from relevant files
    data = np.loadtxt(datafile, dtype = float, delimiter=" ", ndmin=2)
    labels = np.loadtxt(labelfile, dtype = int)
    attributes = np.loadtxt(attributesfile, dtype = str)
    
    # get the best possible tree for our data, from testing we've determined the maximum depth a DecisionTreeClassifier will go for this data is 16
    tree, best, time = get_best_tree(data, labels, depths = np.arange(1, 17), criterions=['entropy'], with_best = True, with_time = True)

    print(f'Best Testing MAE: {best} from tree \n{tree.get_params()}\nwith {tree.get_n_leaves()} leaves and {tree.get_depth()} depth, found in {time} seconds.')

    print(f'Estimated Accuracy: {1 - (best / np.amax(labels))}')

    # Visualize the tree using matplotlib and plot_tree
    #fig = plt.figure(figsize=(100,100))
    #plot_tree(tree, feature_names = attributes, filled = True, fontsize = 20, rounded = True, label=None)
    
    
    #fig.show()
    #fig.savefig("decistion_tree.png")


if __name__ == '__main__':
    main()
