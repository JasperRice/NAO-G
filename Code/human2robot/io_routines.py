import numpy as np
import pandas as pd


def readCSV(filename):
    """Read the CSV file according to the file name and transfer the data frame to a numpy array
    Note that "HUMAN.csv" is defined in degree, while "NAO.csv" is defined in rad
    
    :param filename: The name of the CSV file
    :type filename: string
    :return: The numpy array transferred from the data frame
    :rtype: numpy.ndarray
    """
    df = pd.read_csv(filename)
    array = df.to_numpy()
    return array


def saveNetwork(filename):
    pass


def execute():
    pass


if __name__ == "__main__":
    array = readCSV("HUMAN.csv")
    print array
    print type(array)