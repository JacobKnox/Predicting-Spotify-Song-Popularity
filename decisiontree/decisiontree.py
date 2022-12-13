# decisiontree.py
"""Predict Spotify song popularity based on numerous parameters using a decision tree."""

import numpy as np
import os
import pdb
import sklearn.tree as tree
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
    best_tree, best, time = get_best_tree(data,
                                     labels,
                                     depths = [None],
                                     min_samples_split = [2],
                                     min_samples_leaf = [1],
                                     max_leaf_nodes = [None],
                                     ccp_alpha = [0.0],
                                     criterions = ["entropy"],
                                     with_best = True,
                                     with_time = True,
                                     splitters=["best"],
                                     sample_size = 200)

    clf = best_tree.fit(data[:1499], labels[:1499])
    test = clf.predict(data[1500:])
    num_wrong = sum(test != labels[1500:])
    print(1-(num_wrong/len(test)))
    
    print(f'Best Testing MAE: {best} from tree \n{best_tree.get_params()}\nwith {best_tree.get_n_leaves()} leaves and {best_tree.get_depth()} depth, found in {time} seconds.')

    print(f'Estimated Accuracy: {1 - (best / np.amax(labels))}')

    # Visualize the tree using tree.export_txt
    #text_representation = tree.export_text(best_tree, feature_names=attributes.tolist())
    #with open("decistion_tree.log", "w") as fout:
    #    fout.write(text_representation)


if __name__ == '__main__':
    main()
