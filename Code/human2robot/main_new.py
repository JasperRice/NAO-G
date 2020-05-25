from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('human2robot/')

try: from execute import execGesture
except: pass
from data_processing import decompose, normalize, split, smooth
from Human import HumanInterface
from io_routines import readCSV, saveNetwork
from NAO import NAOInterface
from network import Net, numpy2tensor
from setting import *


if __name__ == "__main__":
    # Reproducibility by setting a seed
    torch.manual_seed(2020)

    human_interface = HumanInterface.createFromBVH('/home/nao/Documents/NAO-G/Code/human2robot/human_skeletion.bvh')
    try: nao_interface = NAOInterface(IP=P_NAO_IP, PORT=P_NAO_PORT)
    except: nao_interface = NAOInterface(IP=NAO_IP, PORT=NAO_PORT)

    # Read dataset
    talk_list = map(readCSV, talkfile)
    talk = np.vstack(talk_list)
    human = readCSV('/home/nao/Documents/NAO-G/Code/human2robot/dataset/Human.csv')
    human_new = readCSV('/home/nao/Documents/NAO-G/Code/human2robot/dataset/Human_new.csv')
    human_new_expand = human_new + np.ra
    human_right_hand = readCSV('/home/nao/Documents/NAO-G/Code/human2robot/dataset/Human_right_hand.csv')

    human_new = np.vstack([human_new, human_new])
    
    nao = readCSV('/home/nao/Documents/NAO-G/Code/human2robot/dataset/NAO.csv')
    nao_new = readCSV('/home/nao/Documents/NAO-G/Code/human2robot/dataset/NAO_new.csv')
    nao_right_hand = readCSV('/home/nao/Documents/NAO-G/Code/human2robot/dataset/NAO_right_hand.csv')
    nao_new = np.vstack([nao_new, nao_new])
    human = np.vstack([human, human_new])
    nao = np.vstack([nao, nao_new])
    n = np.size(human, 0)
    if n != np.size(nao, 0):
        sys.exit("Numbers of input and target are different.")


    # Normalize and decompose the dataset
    if NORMALIZE:
        human, human_scaler = normalize(human)
        nao, nao_scaler = normalize(nao)
        if USE_TALK:
            talk, _ = normalize(talk)

    if USE_TALK:
        _, human_pca = decompose(talk)
        human = human_pca.transform(human)
    else:
        human, human_pca = decompose(human)

    if DECOMPOSE:
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

    
    # Define Neural Network and train

    # net = Net(n_input=np.size(human, 1),
    #             n_hidden=N_HIDDEN,
    #             n_output=np.size(nao, 1),
    #             AF=AF, dropout_rate=DO_RATE
    #             )
    Net.__randomsearch__(human_train_torch, human_val_torch, nao_train_torch, nao_val_torch, max_search=100, filename='/home/nao/Documents/NAO-G/Code/human2robot/dataset/Hyper-parameters.csv')
    exit()
    net = Net.createFromRandomSearch(human_train_torch, human_val_torch, nao_train_torch, nao_val_torch)
    net.__train__(human_train_torch, human_val_torch, nao_train_torch, nao_val_torch)
    net.__plot__()


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

        try: talk_play = human_scaler.transform(talk_play)
        except: pass

        try: talk_play = human_pca.transform(talk_play)
        except: pass
        
        talk_play = numpy2tensor(talk_play)
        prediction = net(talk_play)
        nao_out = prediction.detach().numpy()

        try: nao_out = nao_pca.inverse_transform(nao_out)
        except NameError: pass

        try: nao_out = nao_scaler.inverse_transform(nao_out)
        except NameError: pass

    smooth_kwargs = {
        'window_length':    13,
        'polyorder':        3,
        'deriv':            0,
        'delta':            1.0,
        'axis':             -1,
        'mode':             'interp',
        'cval':             0.0
    }
    nao_out = smooth(nao_out, smoothing_method='savgol', **smooth_kwargs)
    # execGesture(NAO_IP, NAO_PORT, nao_out[:,2:].tolist()) \
    #     if USE_HAND else execGesture(NAO_IP, NAO_PORT, nao_out.tolist())
    to_plot = nao_out.T.tolist()
    plt.plot(to_plot[-1][:100], '-')
    plt.show()