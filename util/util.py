import pickle
from conf.conf import logging

def save_model(dir:str,model)->None:
    logging.info('saving model')
    pickle.dump(model,open(dir,'wb'))
    
def load_model(dir:str)->None:
    logging.info('loading model')
    model = pickle.load(open(dir,'rb'))
    return model