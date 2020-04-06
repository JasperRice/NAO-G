from data_processing import decompose, normalize
from io_routines import execute, readCSV, saveNetwork
from network import Net

import numpy as np
import torch
import torch.nn as nn


if __name__ == "__main__":
    talk_01 = readCSV("dataset/TALK_01.csv")
    talk_02 = readCSV("dataset/TALK_02.csv")
    talk = np.vstack((talk_01, talk_02))
    _, talk_pca = decompose(talk)

    human = readCSV("dataset/HUMAN_Test.csv")
    nao = readCSV("dataset/NAO.csv")
    n, d_human = np.shape(human)
    _, d_nao = np.shape(nao)

    human_n, human_scaler = normalize(human)
    nao_n, nao_scaler = normalize(nao)

    human_n_d = talk_pca



    net = Net(n_input=d_human, n_hidden=250, n_output=d_nao)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = nn.MSELoss()


    for epoch in range(100):
        prediction = net(x)

        loss = loss_func(prediction, )
