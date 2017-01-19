import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

def read_df():
  return pd.read_csv('challenge_dataset.txt', header=None)

def train_test_set():
  data_frame = read_df()
  features = data_frame[[0]]
  target = data_frame[[1]]

  features_train, features_test, target_train, target_test = \
    train_test_split(features, target, test_size=0.5, random_state=42)

  return features, target, features_train, features_test, target_train, target_test


def fit(features, target):
  reg = linear_model.LinearRegression()
  reg.fit(features, target)
  return reg


def print_error(reg, features_train, target_train, features_test, target_test):
  print ("score train: ", reg.score(features_train, target_train))
  print ("score test: ", reg.score(features_test, target_test))  

def predict_and_plot():
  features, target, features_train, features_test, target_train, target_test = train_test_set()
  
  reg = fit(features_train, target_train)

  plt.scatter(features_test, target_test, color='black', label="Training data")
  plt.plot(features_test, reg.predict(features_test), color='blue', label="Prediction")
  plt.legend(loc=2)

  print_error(reg, features_train, target_train, features_test, target_test)

  
  plt.show()

if __name__ == '__main__':
    predict_and_plot()
