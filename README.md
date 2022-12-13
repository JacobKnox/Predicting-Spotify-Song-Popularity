# Predicting Spotify Song Popularity
## Authors
### [Jacob Knox](https://github.com/JacobKnox)
Florida Southern College, Class of 2024
### [Noah Gabryluk](https://github.com/ngabryluk)

## Project
### Description
This repository hosts the code for our CSC 3520 - Machine Learning final project. The purpose of this project is to take data from [Kaggle](https://www.kaggle.com) titled ["Top Hits Spotify from 2000-2019"](https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019) and train various machine learning models to predict song popularity on Spotify.
### Data
While you can download the dataset using the link above, you can find the folder with all the data that we altered the format of to have easier access to [here](https://github.com/JacobKnox/Predicting-Spotify-Song-Popularity/blob/29005cd971e5c22a5399bba4907c35060fd6432a/data). In each row or sample of the [data file](data/data.txt), the values of each attribute are in the same order as they appear in the [attributes file](data/attributes.txt). The [labels file](data/labels.txt) contains all of the popularity ratings for each sample of the data. 
### Required
The list of libraries required to run this code include:
- pip
- numpy
- tensorflow
- keras
- matplotlib
- scikit-learn
### Models
#### [Decision Tree](decisiontree/decisiontree.py)
The decision tree, even after attempting to find the most optimal one, appears to have high accuracy for training data, but extraordinarily low accuracy for novel testing data. Training accuracy appears to hover around 90% of samples with a Mean Absolute Error (MAE) of below 1. Training accuracy, on the other hand, appears to hover around 5% of samples with a Mean Absolute Error (MAE) of 18.
#### [Neural Network](https://github.com/JacobKnox/Predicting-Spotify-Song-Popularity/blob/0db38b88040a81d2f58af5c8cde42aaea5db2296/neuralnetwork/neuralnetwork.py)
Uses Keras models, layers, and optimizers to create and train a neural network on this dataset. We realized that creating this model was more about minimizing the loss rather than having a higher accuracy of correctly predicting the correct popularity rating. This model shows the decrease in loss over time as it is trained.
