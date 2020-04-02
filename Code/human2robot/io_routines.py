# HUMAN.csv is defined in degree
# NAO.csv is defined in rad

import pandas as pd

def readCSV(filename):
    df = pd.read_csv(filename)
    return df

def writeCSV(filename):
    pass

if __name__ == "__main__":
    df = readCSV("NAO.csv")
    a = df.to_numpy()
    print df.head(5)
    print a