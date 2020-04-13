import numpy as np
import pandas as pd
import torch


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

def bvh2CSV(bvh_filename, csv_filename):
    bvh_f = open()


def saveNetwork(filename):
    # torch.save(filename)
    pass


def loadNetwork(filename):
    # return torch.load(filename)
    pass


if __name__ == "__main__":
    array = readCSV("human2robot/dataset/HUMAN.csv")
    print(array)
    print(type(array))