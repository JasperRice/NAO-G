from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('human2robot/')

from constrained_optimization import constraints, objective
try: from execute import execGesture
except: pass
from data_processing import decompose, normalize, split, smooth
from Human import HumanInterface
from io_routines import readCSV, saveNetwork
try: from NAO import NAOInterface
except: pass
from network import Net, numpy2tensor
from setting import *


if __name__ == "__main__":
    # Reproducibility by setting a seed
    torch.manual_seed(2020)

    human_interface = HumanInterface.createFromBVH('dataset/BVH/human_skeletion.bvh')
    try: nao_interface = NAOInterface(IP=P_NAO_IP, PORT=P_NAO_PORT)
    except:
        try: nao_interface = NAOInterface(IP=NAO_IP, PORT=NAO_PORT)
        except: pass

    # Read dataset
    talk_list = map(readCSV, talkfile)
    talk = np.vstack(talk_list)
    human = readCSV('dataset/Human.csv')
    human_new = readCSV('dataset/Human_new.csv')
    human_new_argu = human_new + np.random.normal(loc=0.0, scale=0.6, size=np.shape(human_new))
    human_right_hand = readCSV('dataset/Human_right_hand.csv')
    human_right_hand_argu = human_right_hand + np.random.normal(loc=0.0, scale=0.6, size=np.shape(human_right_hand))
    human = np.vstack([
        # human,
        human_new,
        human_new_argu,
        human_right_hand,
        human_right_hand_argu,
    ])
    
    nao = readCSV('dataset/NAO.csv')
    nao_new = readCSV('dataset/NAO_new.csv')
    nao_new_argu = nao_new + np.random.normal(loc=0.0, scale=0.009, size=np.shape(nao_new))
    nao_right_hand = readCSV('dataset/NAO_right_hand.csv')
    nao_right_hand_argu = nao_right_hand + np.random.normal(loc=0.0, scale=0.009, size=np.shape(nao_right_hand))
    nao = np.vstack([
        # nao,
        nao_new,
        nao_new_argu,
        nao_right_hand,
        nao_right_hand_argu,
    ])

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
    human_test_torch = dataset_torch[2]
    nao_train_torch = dataset_torch[3]
    nao_test_torch = dataset_torch[5]

    # human_val_torch = dataset_torch[1]
    # nao_val_torch = dataset_torch[4]
    human_val_torch = torch.cat(dataset_torch[1:3], dim=0)
    nao_val_torch = torch.cat(dataset_torch[4:6], dim=0)


    # Define Neural Network and train

    # Net.__randomsearch__(human_train_torch, human_val_torch, nao_train_torch, nao_val_torch, max_search=100, filename='dataset/Hyper-parameters-new.csv')
    # net = Net.createFromRandomSearch(human_train_torch, human_val_torch, nao_train_torch, nao_val_torch)
    # exit()

    net = Net(
        n_input=np.size(human, 1),
        n_hidden=N_HIDDEN,
        n_output=np.size(nao, 1),
        AF=AF, dropout_rate=DO_RATE, learning_rate=L_RATE, max_epoch=MAX_EPOCH
    )
    net.__train__(human_train_torch, human_val_torch, nao_train_torch, nao_val_torch)
    net.__plot__()


    # Visualize validation result on NAO
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
        human_interface.readJointAnglesFromBVH('dataset/BVH/NaturalTalking_030_2_1From20.bvh')
        talk_play = human_interface.jointAngles[:100]
        net.eval()

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
        'window_length':    5,
        'polyorder':        3,
        'deriv':            0,
        'delta':            1.0,
        'axis':             -1,
        'mode':             'interp',
        'cval':             0.0
    }

    # =====> Plot before smoothing
    # to_plot = nao_out.T.tolist()

    # plt.plot(to_plot[6][:100], '-', c='blue') # LWristYaw
    # plt.xlabel('Timestamp')
    # plt.ylabel('Joint angle / rad')
    # plt.legend(['LWristYaw'])
    # plt.show()
    # plt.plot(to_plot[-1][:100], '-', c='blue') # RWristYaw
    # plt.xlabel('Timestamp')
    # plt.ylabel('Joint angle / rad')
    # plt.legend(['RWristYaw'])
    # plt.show()
    # plt.plot(to_plot[2][:100], '-', c='blue') # LShoulderPitch
    # plt.xlabel('Timestamp')
    # plt.ylabel('Joint angle / rad')
    # plt.legend(['LShoulderPitch'])
    # plt.show()
    # plt.plot(to_plot[-5][:100], '-', c='blue') # RShoulderPitch
    # plt.xlabel('Timestamp')
    # plt.ylabel('Joint angle / rad')
    # plt.legend(['RShoulderPitch'])
    # plt.show()


    # =====> Smoothing
    nao_out = smooth(nao_out, smoothing_method='savgol', **smooth_kwargs)


    # =====> Plot before smoothing
    # to_plot = nao_out.T.tolist()

    # plt.plot(to_plot[6][:100], '-', c='blue') # LWristYaw
    # plt.xlabel('Timestamp')
    # plt.ylabel('Joint angle / rad')
    # plt.legend(['LWristYaw'])
    # plt.show()
    # plt.plot(to_plot[-1][:100], '-', c='blue') # RWristYaw
    # plt.xlabel('Timestamp')
    # plt.ylabel('Joint angle / rad')
    # plt.legend(['RWristYaw'])
    # plt.show()
    # plt.plot(to_plot[2][:100], '-', c='blue') # LShoulderPitch
    # plt.xlabel('Timestamp')
    # plt.ylabel('Joint angle / rad')
    # plt.legend(['LShoulderPitch'])
    # plt.show()
    # plt.plot(to_plot[-5][:100], '-', c='blue') # RShoulderPitch
    # plt.xlabel('Timestamp')
    # plt.ylabel('Joint angle / rad')
    # plt.legend(['RShoulderPitch'])
    # plt.show()
    # exit()

    
    all_limits = nao_interface.limits
    h = 1.0 / 60 * 20
    nao_out = nao_out.T
    for i, x in enumerate(nao_out):
        print("Optimizing joint {}.".format(i))
        x0 = x
        limits = {
            'minAngle':     all_limits['minAngle'][i],
            'maxAngle':     all_limits['maxAngle'][i],
            'maxChange':    all_limits['maxChange'][i]
        }
        cons = constraints(x, limits, h)
        args = (np.eye(np.size(x)) * 5.0,
                np.eye(np.size(x)-1) * 0.1,
                h,
                x
               )
        sol = minimize(objective, x0, args=args, method='SLSQP', constraints=cons, options={'disp': True})
        nao_out[i] = sol.x
    nao_out = nao_out.T

    # =====> Plot after constrained optimization
    # to_plot = nao_out.T.tolist()

    # plt.plot(to_plot[6][:100], '-', c='blue') # LWristYaw
    # plt.xlabel('Timestamp')
    # plt.ylabel('Joint angle / rad')
    # plt.legend(['LWristYaw'])
    # plt.show()
    # plt.plot(to_plot[-1][:100], '-', c='blue') # RWristYaw
    # plt.xlabel('Timestamp')
    # plt.ylabel('Joint angle / rad')
    # plt.legend(['RWristYaw'])
    # plt.show()
    # plt.plot(to_plot[2][:100], '-', c='blue') # LShoulderPitch
    # plt.xlabel('Timestamp')
    # plt.ylabel('Joint angle / rad')
    # plt.legend(['LShoulderPitch'])
    # plt.show()

    # =====> Execute the motion
    raw_input("Press ENTER to execute the motion.")
    execGesture(NAO_IP, NAO_PORT, nao_out.tolist(), TIME=h)