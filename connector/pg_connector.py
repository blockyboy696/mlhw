import pandas as pd
from conf.conf import logging

def get_data(link: str) -> pd.DataFrame:
    logging.info('extractinng data')
    df = pd.read_csv(link)
    logging.info('data extracted & df created')
    return df 