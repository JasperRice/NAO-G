from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('human2robot/')
try:
    from execute import execGesture
except:
    pass
from data_processing import decompose, normalize, split, smooth
from io_routines import readCSV, saveNetwork
from NAO import Interface
from network import Net, numpy2tensor
from setting import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


if __name__ == "__main__":
    # Reproducibility by setting a seed
    torch.manual_seed(0)


    # Read dataset
    if USE_TALK:
        talk_list = map(readCSV, filename[:-2])
        talk = np.vstack(talk_list)
    human = readCSV(filename[-2])
    nao = readCSV(filename[-1]) if USE_HAND else readCSV(filename[-1])[:,2:]
    # Test gestures
    # execGesture(NAO_IP, NAO_PORT, nao[:,2:].tolist()) \
            # if USE_HAND else execGesture(NAO_IP, NAO_PORT, nao.tolist())
    
    n = np.size(human, 0)
    if n != np.size(nao, 0):
        sys.exit("Numbers of input and target are different.")


    # Normalize and decompose the dataset
    if NORMALIZE:
        human, human_scaler = normalize(human)
        nao, nao_scaler = normalize(nao)
        if USE_TALK:
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


    # Save the shuffled train, test, and validation data and visualize the ground truth of the NAO gestures
    if SAVE_DATA:
        human_to_save = dataset[0:3]
        nao_to_save = dataset[3:]
        try:
            human_to_save = map(human_pca.inverse_transform, human_to_save)
            nao_to_save = map(nao_pca.inverse_transform, nao_to_save)
        except NameError:
            pass
        try:
            human_to_save = map(human_scaler.inverse_transform, human_to_save)
            nao_to_save = map(nao_scaler.inverse_transform, nao_to_save)
        except NameError:
            pass
        np.savetxt("human_train.txt", human_to_save[0])
        np.savetxt("human_val.txt", human_to_save[1])
        np.savetxt("human_test.txt", human_to_save[2])
        np.savetxt("nao_train.txt", nao_to_save[0])
        np.savetxt("nao_val.txt", nao_to_save[1])
        np.savetxt("nao_test.txt", nao_to_save[2])
        execGesture(NAO_IP, NAO_PORT, nao_to_save[ON_SET][:,2:].tolist()) \
            if USE_HAND else execGesture(NAO_IP, NAO_PORT, nao_to_save[ON_SET].tolist())
        exit()


    # Transfer the numpy to tensor in pytorch
    dataset_torch = map(numpy2tensor, dataset)
    human_train_torch = dataset_torch[0]
    human_val_torch = dataset_torch[1]
    human_test_torch = dataset_torch[2]
    nao_train_torch = dataset_torch[3]
    nao_val_torch = dataset_torch[4]
    nao_test_torch = dataset_torch[5]

    # nao_interface = Interface(NAO_IP, NAO_PORT)
    # Define Neural Network and train
    net = Net(n_input=np.size(human, 1),
                n_hidden=N_HIDDEN,
                n_output=np.size(nao, 1),
                AF=AF, dropout_rate=DO_RATE
                )
    # net = Net(n_input=np.size(human, 1), n_hidden=N_HIDDEN, n_output=np.size(nao, 1), AF=AF, dropout_rate=DO_RATE)
    optimizer = torch.optim.SGD(net.parameters(), lr=L_RATE)
    loss_func = nn.MSELoss()


    # Main loop for training
    min_error = np.inf
    for epoch in range(MAX_EPOCH):
        print("=====> Epoch: "+str(epoch+1))

        # Train
        net.train()
        prediction = net(human_train_torch)
        loss = loss_func(prediction, nao_train_torch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation error
        net.eval()
        val_err = loss_func(net(human_val_torch), nao_val_torch)
        
        if STOP_EARLY:
            if val_err - min_error > STOP_RATE * min_error:
                break
            elif val_err < min_error:
                min_error = val_err

        # Plot the error
        plt.scatter(epoch, loss.data.numpy(), s=1, c='r')
        plt.scatter(epoch, val_err.data.numpy(), s=1, c='g')

    plt.show()
    plt.savefig(AF+'_LR='+str(L_RATE)+'_Hidden='+str(N_HIDDEN)+'_Normalize='+str(NORMALIZE)+'_Decompose='+str(DECOMPOSE)+'_Dropout='+str(DO_RATE)+'.png')


    # Visualize result on NAO
    if VISUALIZE:
        net.eval()
        prediction = net(dataset_torch[ON_SET])
        nao_out = prediction.detach().numpy()
        try:
            nao_out = nao_pca.inverse_transform(nao_out)
        except NameError:
            pass
        try:
            nao_out = nao_scaler.inverse_transform(nao_out)
        except NameError:
            pass
        execGesture(NAO_IP, NAO_PORT, nao_out[:,2:].tolist()) \
            if USE_HAND else execGesture(NAO_IP, NAO_PORT, nao_out.tolist())


    # Play the talk
    if PLAY_TALK:
        net.eval()
        talk_play = readCSV(filename[PLAY_SET])
        try:
            talk_play = human_scaler.transform(talk_play)
        except:
            pass

        try:
            talk_play = human_pca.transform(talk_play)
        except:
            pass
        
        talk_play = numpy2tensor(talk_play)
        prediction = net(talk_play)
        nao_out = prediction.detach().numpy()
        try:
            nao_out = nao_pca.inverse_transform(nao_out)
        except NameError:
            pass
        try:
            nao_out = nao_scaler.inverse_transform(nao_out)
        except NameError:
            pass

        kwargs = {
            'window_length':    49,
            'polyorder':        5,
            'deriv':            0,
            'delta':            1.0,
            'axis':             -1,
            'mode':             'interp',
            'cval':             0.0
        }
        nao_out = smooth(nao_out, smoothing_method='savgol', **kwargs)
        execGesture(NAO_IP, NAO_PORT, nao_out[:,2:].tolist()) \
            if USE_HAND else execGesture(NAO_IP, NAO_PORT, nao_out.tolist())