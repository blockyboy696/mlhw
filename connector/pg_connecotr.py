import pandas as pd
def get_data(link: str) -> pd.DataFrame:
    #extract data from the link
    df = pd.read_csv(link)
    return df