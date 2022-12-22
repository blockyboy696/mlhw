import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from connector.pg_connecotr import get_data
from conf.conf import logging

logging.info("extracting dataset")
df = get_data('https://raw.githubusercontent.com/5x12/ml-cookbook/master/supplements/data/heart.csv')
logging.info('extracting confirmed')

def train_test_split(df):
    #setting independent variables and dependent variable
    X = df.iloc[:, :-1]
    y = df['target']
    #spliiting the data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, #independent variables
                                                    y, #dependent variable
                                                    random_state = 3
                                                  )
    return X_train, X_test, y_train, y_test

def trainining_naive_bayes(X_train, y_train):
    #initializing the model
    clf = GaussianNB()
    #training the model
    clf.fit(X_train, y_train)
    return clf
