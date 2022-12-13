# perceptron.py
# Train a simple perceptron to classify letters.

import argparse
import numpy as np
import os

ROOT = os.path.dirname(os.path.abspath(__file__))  # root directory of this code


parser = argparse.ArgumentParser(description="Parse input from Kaggle CSV data for song popularity prediction.")
parser.add_argument('-x', '--data',
                    help='Path to the desired data file, defaults to ROOT/songs_normalize.csv',
                    default=os.path.join(ROOT, 'songs_normalize.csv'))
parser.add_argument('-y', '--labels',
                    help='Path to file where labels will be saved, defaults to ROOT/labels.txt',
                    default=os.path.join(ROOT, 'labels.txt'))
parser.add_argument('-s', '--save',
                    help='Path to file where parsed data will be saved, defaults to ROOT/data.txt',
                    default=os.path.join(ROOT, 'data.txt'))
parser.add_argument('-a', '--attributes',
                    help='Path to file where attributes will be saved, defaults to ROOT/data.txt',
                    default=os.path.join(ROOT, 'attributes.txt'))


def main(args):
    # Parse input arguments
    datafile = os.path.expanduser(args.data)
    labelfile = os.path.expanduser(args.labels)
    attributefile = os.path.expanduser(args.attributes)
    savefile = os.path.expanduser(args.save)

    data = np.loadtxt(datafile, dtype=str, delimiter=',', encoding="utf8", usecols=np.arange(2,16)) # load in the data, ignoring the artist and song name as well as the genres
    attributes = np.delete(data[0], 3) # get the attributes, ignoring the popularity one
    data = data[1:] # get rid of the first row in the data, which is the attributes
    labels = data[:, 3] # get the labels
    data = np.delete(data, 3, 1) # delete the third column in the data, which is the labels
    data[:, 1] = (data[:, 1] == 'True').astype(int) # converts 'True'/'False' under 'explicit' to 1/0

    np.savetxt(attributefile, attributes, fmt='%s')
    np.savetxt(labelfile, labels, fmt='%s')
    np.savetxt(savefile, data, fmt='%s')
    

if __name__ == '__main__':
    main(parser.parse_args())
