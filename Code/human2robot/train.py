from data_processing import decompose, normalize
from io_routines import execute, readCSV, saveNetwork
from network import Net

import numpy as np


if __name__ == "__main__":
    talk_01 = readCSV("dataset/TALK_01.csv")
    talk_02 = readCSV("dataset/TALK_02.csv")
    talk = np.vstack((talk_01, talk_02))

    human = readCSV("dataset/HUMAN.csv")
    nao = readCSV("dataset/NAO.csv")



