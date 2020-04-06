from data_processing import decompose, normalize
from io_routines import execute, readCSV, saveNetwork
from network import Net

import numpy as np
import torch
import torch.nn as nn

def train(n_hidden):
    pass


if __name__ == "__main__":
    talk_01 = readCSV("dataset/TALK_01.csv")
    talk_02 = readCSV("dataset/TALK_02.csv")
    talk = np.vstack((talk_01, talk_02))

    human = readCSV("dataset/HUMAN.csv")
    nao = readCSV("dataset/NAO.csv")

    n, d_human = np.shape(human)
    _, d_nao = np.shape(nao)

    net = Net(n_input=d_human, n_hidden=250, n_output=d_nao)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = nn.MSELoss()
