from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_processing import decompose, normalize, split
from execute import execGesture
from io_routines import readCSV, saveNetwork
from network import Net

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(0)

    # Whole data "talk"
    talk_01 = readCSV("human2robot/dataset/TALK_01.csv")
    talk_02 = readCSV("human2robot/dataset/TALK_02.csv")
    talk = np.vstack((talk_01, talk_02))

    # Normalize and decompose the dataset
    human = readCSV("human2robot/dataset/HUMAN.csv")
    nao = readCSV("human2robot/dataset/NAO.csv")
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
    
    # Save the shuffled pose for visualization
    # np.savetxt("npSaveTest.txt", human_scaler.inverse_transform(talk_pca.inverse_transform(human_train)))

    # Transfer the numpy to tensor in pytorch
    human_train_torch = torch.from_numpy(human_train).float()
    human_val_torch = torch.from_numpy(human_val).float()
    human_test_torch = torch.from_numpy(human_test).float()
    nao_train_torch = torch.from_numpy(nao_train).float()
    nao_val_torch = torch.from_numpy(nao_val).float()
    nao_test_torch = torch.from_numpy(nao_test).float()

    # Define Neural Network and train
    net = Net(n_input=talk_pca.n_components_, n_hidden=250, n_output=nao_pca.n_components_)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = nn.MSELoss()

    # Main loop for training
    old_val_err = 1000
    for epoch in range(300):
        print("=====> Epoch: "+str(epoch+1))

        # Train
        prediction = net(human_train_torch)
        loss = loss_func(prediction, nao_train_torch)
        val_err = loss_func(net(human_val_torch), nao_val_torch)
        
        if val_err > old_val_err:
            break
        old_val_err = val_err

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot the error
        plt.scatter(epoch, loss.data.numpy(), s=1, c='r')
        plt.scatter(epoch, val_err.data.numpy(), s=1, c='g')

    plt.show()

    # Visualize train result on NAO
    nao_out = prediction.detach().numpy()
    nao_out = nao_pca.inverse_transform(nao_out)
    nao_out = nao_scaler.inverse_transform(nao_out)
    nao_out = nao_out.tolist()
    # execGesture("127.0.0.1", 45817, nao_out[50][2:])

    # Visualize validation result on NAO
    prediction = net(human_val_torch)
    nao_out = prediction.detach().numpy()
    nao_out = nao_pca.inverse_transform(nao_out)
    nao_out = nao_scaler.inverse_transform(nao_out)
    nao_out = nao_out.tolist()
    execGesture("127.0.0.1", 45817, nao_out[5][2:])
