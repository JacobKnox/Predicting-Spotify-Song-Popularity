# Predicting Spotify Song Popularity
## Authors
### [Jacob Knox](https://github.com/JacobKnox)
Florida Southern College, Class of 2024
### [Noah Gabryluk](https://github.com/ngabryluk)

## Project
### Description
This repository hosts the code for our CSC 3520 - Machine Learning final project. The purpose of this project is to take data from [Kaggle](https://www.kaggle.com) titled ["Top Hits Spotify from 2000-2019"](https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019) and train various machine learning models to predict song popularity on Spotify.
### Models
#### [Decision Tree](decisiontree/decisiontree.py)
The decision tree, even after attempting to find the most optimal one, appears to have high accuracy for training data, but extraordinarily low accuracy for novel testing data. Training accuracy appears to hover around 90% of samples with a Mean Absolute Error (MAE) of below 1. Training accuracy, on the other hand, appears to hover around 5% of samples with a Mean Absolute Error (MAE) of 18.
#### [Neural Network](https://github.com/JacobKnox/Predicting-Spotify-Song-Popularity/blob/0db38b88040a81d2f58af5c8cde42aaea5db2296/neuralnetwork/neuralnetwork.py)
Uses a Keras Sequential Class to create and train a neural network on this dataset.
