from conf.conf import logging, settings
from models.model import split,training_model,predict
from connector.pg_connector import get_data
from util.util import load_model
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from models.boosting import gridsearch_nb, gridsearch_lr
import argparse

#parser = argparse.ArgumentParser()
#args = parser.parse_args()
#parser.add_argument('x')

settings.load_file(path="conf/settings.toml")

df = get_data(settings.DATA.data_set)

X_train, X_test, y_train, y_test = split(df)

nb_params = gridsearch_nb(X_train, y_train)

nb = training_model(GaussianNB, X_train, y_train,**nb_params)

lr_params = gridsearch_lr(X_train, y_train)

lr = training_model(LogisticRegression, X_train, y_train, **lr_params)

prediction_data = [[59, 1, 0, 101, 234, 0, 1, 143, 0, 3.4, 0, 0, 0]]

predict(GaussianNB,prediction_data)
predict(LogisticRegression,prediction_data)