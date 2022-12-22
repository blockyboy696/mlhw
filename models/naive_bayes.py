import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from connector.pg_connector import get_data
from conf.conf import logging

df = get_data('https://raw.githubusercontent.com/5x12/ml-cookbook/master/supplements/data/heart.csv')

def train_test_split(df: pd.DataFrame)-> pd.DataFrame:
    logging.info('spliinug the df to X and y')
    X = df.iloc[:, :-1]
    y = df['target']
    logging.info('spliiting the data to train and test')
    X_train, X_test, y_train, y_test = train_test_split(X, #independent variables
                                                    y, #dependent variable
                                                    random_state = 3
                                                  )
    return X_train, X_test, y_train, y_test

def trainining_naive_bayes(X_train, y_train):
    logging.info('initializing the model')
    clf = GaussianNB()
    logging.info('training the model')
    clf.fit(X_train, y_train)
    return clf
