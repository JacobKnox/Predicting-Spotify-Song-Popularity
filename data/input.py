# perceptron.py
# Train a simple perceptron to classify letters.

import argparse
import numpy as np
import os

ROOT = os.path.dirname(os.path.abspath(__file__))  # root directory of this code


parser = argparse.ArgumentParser(description="Parse input from Kaggle CSV data for song popularity prediction.")
parser.add_argument('-x', '--data',
                    help='Path to the desired data file, defaults to ROOT/data.txt',
                    default=os.path.join(ROOT, 'data.txt'))
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

    data = np.loadtxt(datafile, dtype=str, delimiter=',', encoding="utf8", usecols=np.arange(2,16))
    attributes = np.delete(data[0], 5)
    data = data[1:]
    labels = data[:, 3]

    np.savetxt(attributefile, attributes, fmt='%s')
    np.savetxt(labelfile, labels, fmt='%s')
    np.savetxt(savefile, data, fmt='%s')
    

if __name__ == '__main__':
    main(parser.parse_args())
