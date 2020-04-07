from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_processing import decompose, normalize, split
from io_routines import execute, readCSV, saveNetwork
from network import Net

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


if __name__ == "__main__":
    # Whole data "talk"
    talk_01 = readCSV("dataset/TALK_01.csv")
    talk_02 = readCSV("dataset/TALK_02.csv")
    talk = np.vstack((talk_01, talk_02))

    # Normalize and decompose the dataset
    human = readCSV("dataset/HUMAN_Test.csv")
    nao = readCSV("dataset/NAO.csv")
    n_human, _ = np.shape(human)
    n_nao, _ = np.shape(nao)
    if n_human is not n_nao:
        print("Number of input and target are different")
        exit()
    n = n_human

    talk_n, _ = normalize(talk)
    _, talk_pca = decompose(talk_n)
    human_n, human_scaler = normalize(human)
    human_n_d = talk_pca.transform(human_n)
    
    nao_n, nao_scaler = normalize(nao)
    nao_n_d, nao_pca = decompose(nao_n)

    # Split the dataset into train, test, and validation
    human_train, human_test, human_val, nao_train, nao_test, nao_val = split(human_n_d, nao_n_d)
    
    # Transfer the numpy to tensor in pytorch
    human_train_torch = torch.from_numpy(human_train).float()
    human_val_torch = torch.from_numpy(human_val).float()
    # human_test_torch = torch.double(torch.from_numpy(human_test))
    nao_train_torch = torch.from_numpy(nao_train).float()
    nao_val_torch = torch.from_numpy(nao_val).float()
    # nao_test_torch = torch.double(torch.from_numpy(nao_test))


    if False:
        print(talk_pca.n_components_)
        print(nao_pca.n_components_)
        print(np.shape(human))
        print(np.shape(human_n_d))
        print(np.shape(human_train))
        print(np.shape(human_test))
        print(np.shape(human_val))
        print(np.shape(nao))
        print(np.shape(nao_n_d))
        print(np.shape(nao_train))
        print(np.shape(nao_test))
        print(np.shape(nao_val))
        exit()

    # Define Neural Network and train
    net = Net(n_input=talk_pca.n_components_, n_hidden=250, n_output=nao_pca.n_components_)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = nn.MSELoss()

    # Main loop for training
    plt.figure()
    for epoch in range(100):
        print("=====> Epoch: "+str(epoch+1))

        # Train
        prediction = net(human_train_torch)
        loss = loss_func(prediction, nao_train_torch)
        val_err = loss_func(net(human_val_torch), nao_val_torch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot the error
        plt.scatter(epoch, loss.data.numpy(), s=1, c='r')
        plt.scatter(epoch, val_err.data.numpy(), s=1, c='g')

    plt.show()