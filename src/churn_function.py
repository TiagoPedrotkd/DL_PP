import pandas as pd

def import_data(filePath):
    df = pd.read_csv(filePath)

    return df
