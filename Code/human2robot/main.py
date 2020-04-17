from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('human2robot/')
try:
    from execute import execGesture
except:
    pass
from data_processing import decompose, normalize, split
from io_routines import readCSV, saveNetwork
from network import Net, numpy2tensor

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


if __name__ == "__main__":
    NAO_IP      = "127.0.0.1"
    NAO_PORT    = 43767
    VISUALIZE   = True
    ON_SET      = 0 # Visualize on 0: train, 1: validation or 3: test
    NORMALIZE   = True
    DECOMPOSE   = True
    USE_TALK    = True
    USE_HAND    = False
    STOP_EARLY  = False
    SAVE_DATA   = False
    MAX_EPOCH   = 300
    N_HIDDEN    = 64
    DO_RATE     = 0.0
    AF          = 'relu'
    PATHNAME    = "human2robot/dataset/"
    FILENAME    = ["TALK_01.csv", "TALK_02.csv", #"TALK_04.csv", "TALK_05.csv",
                   "HUMAN.csv",
                   "NAO.csv"]
    filename    = map(lambda x: PATHNAME + x, FILENAME)
    
    # Reproducibility
    torch.manual_seed(0)

    # Read dataset
    if USE_TALK:
        talk_list = map(readCSV, filename[:-2])
        talk = np.vstack(talk_list)

    human = readCSV(filename[-2])
    nao = readCSV(filename[-1]) if USE_HAND else readCSV(filename[-1])[:,2:]
    n = np.size(human, 0)
    if n != np.size(nao, 0):
        sys.exit("Numbers of input and target are different.")

    # Normalize and decompose the dataset
    if NORMALIZE:
        human, human_scaler = normalize(human)
        nao, nao_scaler = normalize(nao)
        talk, _ = normalize(talk)

    if DECOMPOSE:
        if USE_TALK:
            _, human_pca = decompose(talk)
            human = human_pca.transform(human)
        else:
            human, human_pca = decompose(human)

        nao, nao_pca = decompose(nao)

    # Split the dataset into train, test, and validation
    dataset = split(human, nao)

    # Save the shuffled pose for visualization
    if SAVE_DATA:
        np.savetxt("human_train.txt", human_scaler.inverse_transform(human_pca.inverse_transform(dataset[0])))
        np.savetxt("human_val.txt", human_scaler.inverse_transform(human_pca.inverse_transform(dataset[1])))
        np.savetxt("human_test.txt", human_scaler.inverse_transform(human_pca.inverse_transform(dataset[2])))
        exit()
    
   # Transfer the numpy to tensor in pytorch
    dataset_torch = map(numpy2tensor, dataset)
    human_train_torch = dataset_torch[0]
    human_val_torch = dataset_torch[1]
    human_test_torch = dataset_torch[2]
    nao_train_torch = dataset_torch[3]
    nao_val_torch = dataset_torch[4]
    nao_test_torch = dataset_torch[5]

    # Define Neural Network and train
    net = Net(n_input=human_pca.n_components_, n_hidden=N_HIDDEN, n_output=nao_pca.n_components_, AF=AF, dropout_rate=DO_RATE)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = nn.MSELoss()

    # Main loop for training
    old_val_err = 1000
    for epoch in range(MAX_EPOCH):
        print("=====> Epoch: "+str(epoch+1))

        # Train
        prediction = net(human_train_torch)
        loss = loss_func(prediction, nao_train_torch)
        val_err = loss_func(net(human_val_torch), nao_val_torch)
        
        if STOP_EARLY and val_err > old_val_err:
            break
        old_val_err = val_err

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot the error
        plt.scatter(epoch, loss.data.numpy(), s=1, c='r')
        plt.scatter(epoch, val_err.data.numpy(), s=1, c='g')

    plt.show()

    # Visualize result on NAO
    if VISUALIZE:
        prediction = net(dataset_torch[ON_SET])
        nao_out = prediction.detach().numpy()
        try:
            nao_out = nao_pca.inverse_transform(nao_out)
        except:
            pass

        try:
            nao_out = nao_scaler.inverse_transform(nao_out)
        except:
            pass
        
        execGesture(NAO_IP, NAO_PORT, nao_out[:,2:].tolist()) \
            if USE_HAND else execGesture(NAO_IP, NAO_PORT, nao_out.tolist())
