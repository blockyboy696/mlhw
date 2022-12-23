from conf.conf import logging, settings
from models.naive_bayes import split,trainining_naive_bayes, naive_bayes_score
from connector.pg_connector import get_data
from util.util import load_model

settings.load_file(path="conf/settings.toml")
df = get_data(settings.DATA.data_set)
dir = settings.DIR.dir

X_train, X_test, y_train, y_test = split(df)

trainining_naive_bayes(X_train,y_train)
naive_bayes_score(dir, X_test,y_test)