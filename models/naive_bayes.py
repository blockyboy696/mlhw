import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


from connector.pg_connector import get_data
from conf.conf import logging, settings
from util.util import save_model,load_model
import dynaconf

def split(df: pd.DataFrame)-> pd.DataFrame:
    logging.info('spliinug the df to X and y')
    X = df.iloc[:, :-1]
    y = df['target']
    logging.info('spliiting the data to train and test')
    X_train, X_test, y_train, y_test = train_test_split(X, #independent variables
                                                    y, #dependent variable
                                                    random_state = 42
                                                  )
    return X_train, X_test, y_train, y_test

def trainining_naive_bayes(X_train: pd.DataFrame, y_train: pd.DataFrame):
    settings.load_file(path="conf/settings.toml")
    logging.info('initializing the model')
    clf = GaussianNB()
    logging.info('training the model')
    clf.fit(X_train, y_train)
    save_model(settings.Dir.dir, clf)
    return clf

def naive_bayes_score(dir, X_test, y_test):
    
    logging.info('loading model')
    clf = load_model(dir)
    logging.info(f"model score is {clf.score(X_test, y_test)}")
