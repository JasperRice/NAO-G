from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn

import sys
sys.path.append('human2robot/')

from constrained_optimization import constraints, objective, piecewise_constraints, piecewise_objective
try: from execute import execGesture
except: pass
from data_processing import decompose, normalize, split, smooth
from Human import HumanInterface
from io_routines import readCSV, saveNetwork
try: from NAO import NAOInterface
except: pass
from network import Net, numpy2tensor
from setting import *

def plot_joint_sequence(joints, jointNames, sequence, limits, h, filename=None):
    for joint in joints:
        n = len(sequence[joint])
        angles = sequence[joint]
        vel = [(angles[i+1] - angles[i]) / h for i in range(n-1)]

        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(angles, '-', c='blue')
        axs[0].hlines(y=[limits['minAngle'][joint], limits['maxAngle'][joint]],
                      xmin=0, xmax=n-1, linestyles='dashed')
        axs[0].xlabel('Timestamp')
        axs[0].ylabel('Joint angle / rad')
        axs[0].legend([jointNames[joint]])

        axs[1].plot(vel, '-', c='blue')
        plt.hlines(y=[limits['maxChange'][joint], -limits['maxChange'][joint]],
                    xmin=0, xmax=n-2, linestyles='dashed')
        axs[1].xlabel('Timestamp')
        axs[1].ylabel('Joint angular velocity / rad/s')
        axs[1].legend([jointNames[joint]])
        if filename: pass
        else: plt.show()

def jerk(motions, f):
    joints_sequence = motions.T
    J = 0.0
    for S in joints_sequence:
        N = len(S)
        for i in range(N-3):
            J += ((S[i+3] - 3.0*S[i+2] + 3.0*S[i+1] - S[i]) * (f**3))**2
    return J / (2*f*(N-3))

if __name__ == "__main__":
    # Reproducibility by setting a seed
    torch.manual_seed(0)

    # Create interface for Human and NAO
    # Human interface is used for recording the finger indices
    human_interface = HumanInterface.createFromBVH('dataset/BVH/human_skeletion.bvh')
    try: nao_interface = NAOInterface(IP=P_NAO_IP, PORT=P_NAO_PORT)
    except:
        try: nao_interface = NAOInterface(IP=NAO_IP, PORT=NAO_PORT)
        except: pass
    fingerIndex = []
    fingerJointList = [
            'RightHandThumb1',   'RightHandThumb2',     'RightHandThumb3',
            'RightHandIndex1',   'RightHandIndex2',     'RightHandIndex3',
            'RightHandMiddle1',  'RightHandMiddle2',    'RightHandMiddle3',
            'RightHandRing1',    'RightHandRing2',      'RightHandRing3',
            'RightHandPinky1',   'RightHandPinky2',     'RightHandPinky3',
            'LeftHandThumb1',    'LeftHandThumb2',      'LeftHandThumb3',
            'LeftHandIndex1',    'LeftHandIndex2',      'LeftHandIndex3',
            'LeftHandMiddle1',   'LeftHandMiddle2',     'LeftHandMiddle3',
            'LeftHandRing1',     'LeftHandRing2',       'LeftHandRing3',
            'LeftHandPinky1',    'LeftHandPinky2',      'LeftHandPinky3',
        ]
    for fingerJoint in fingerJointList:
        index = human_interface.getStartAngleIndex(fingerJoint)
        fingerIndex.extend([i for i in range(index, index+3)])
    fingerIndex.extend([0, 1, 2])

    # Read traing and test dataset
    human = readCSV('dataset/Human_overlap.csv'); human = np.delete(human, fingerIndex, axis=1)
    human_test = readCSV('dataset/Human_test.csv'); human_test = np.delete(human_test, fingerIndex, axis=1)
    nao = readCSV('dataset/NAO_overlap.csv')
    nao_test = readCSV('dataset/NAO_test.csv')
    nao_test_copy = nao_test
    if np.size(human, 0) != np.size(nao, 0):
        sys.exit("Numbers of input and target are different.")

    # Normalize and decompose the dataset
    human, human_scaler = normalize(human)
    human_test = human_scaler.transform(human_test)
    nao, nao_scaler = normalize(nao)
    nao_test = nao_scaler.transform(nao_test)

    # Split the dataset into train, test, and validation
    human_train, human_val, nao_train, nao_val = train_test_split(human, nao, test_size=0.2, random_state=1000)
    human_train_torch   = torch.from_numpy(human_train).float()
    human_val_torch     = torch.from_numpy(human_val).float()
    human_test_torch    = torch.from_numpy(human_test).float()
    nao_train_torch     = torch.from_numpy(nao_train).float()
    nao_val_torch       = torch.from_numpy(nao_val).float()
    nao_test_torch      = torch.from_numpy(nao_test).float()

    # Save the shuffled train, validation data
    human_train_save = human_scaler.inverse_transform(human_train); np.savetxt("shuffled_human_train.txt", human_train_save)
    human_val_save = human_scaler.inverse_transform(human_val);     np.savetxt("shuffled_human_val.txt", human_val_save)
    human_test_save = human_scaler.inverse_transform(human_test);   np.savetxt("shuffled_human_test.txt", human_test_save)
    nao_train_save = nao_scaler.inverse_transform(nao_train);       np.savetxt("shuffled_nao_train.txt", nao_train_save)
    nao_val_save = nao_scaler.inverse_transform(nao_val);           np.savetxt("shuffled_nao_val.txt", nao_val_save)
    nao_test_save = nao_scaler.inverse_transform(nao_test);         np.savetxt("shuffled_nao_test.txt", nao_test_save)

    # Define Neural Network model and train
    net = Net(n_input=np.size(human, 1), n_hidden=[128, 32], n_output=np.size(nao, 1),
              AF='relu', dropout_rate=0.0765078199812, learning_rate=0.0002296913506475621, reg=0.005450020325607934, ues_lr_scheduler=False)
    net.__train__(human_train_torch, human_val_torch, nao_train_torch, nao_val_torch, max_epoch=5000, stop=True, scaler=nao_scaler)
    net.__plot__(denormalized=True)
    save_model = True
    if save_model: torch.save(net, "net.txt")

    net.eval()
    
    # Evaluate the Test Error
    nao_test_out = net(human_test_torch).detach().numpy()
    try: nao_test_out = nao_pca.inverse_transform(nao_test_out)
    except NameError: pass
    try: nao_test_out = nao_scaler.inverse_transform(nao_test_out)
    except NameError: pass
    print("RMSE for test data: {} rad.".format(net.loss_func(torch.from_numpy(nao_test_out).float(), torch.from_numpy(nao_test_copy).float()).item()))

    Execute_Test_on_NAO = False
    if Execute_Test_on_NAO:
        # Execute test results on NAO
        raw_input("Press ENTER to execute the test results.")
        try: execGesture(P_NAO_IP, P_NAO_PORT, nao_test_out.tolist())
        except: execGesture(NAO_IP, NAO_PORT, nao_test_out.tolist())

        # Execute test data (ground truth) on NAO
        try: nao_test = nao_pca.inverse_transform(nao_test)
        except NameError: pass
        try: nao_test = nao_scaler.inverse_transform(nao_test)
        except NameError: pass
        raw_input("Press ENTER to execute the ground truth.")
        try: execGesture(P_NAO_IP, P_NAO_PORT, nao_test.tolist())
        except: execGesture(NAO_IP, NAO_PORT, nao_test.tolist())

    # Plot motion sequence on joints
    human_interface.readJointAnglesFromBVH('dataset/BVH/NaturalTalking_030_2_1From5.bvh')
    h = 1.0 / 60.0 * 5.0
    talk_play = human_interface.jointAngles[:500]                
    talk_play = np.delete(np.array(talk_play), fingerIndex, axis=1)
    try: talk_play = human_scaler.transform(talk_play)
    except NameError: pass
    try: talk_play = human_pca.transform(talk_play)
    except NameError: pass
    talk_play_torch = torch.from_numpy(talk_play).float()
    talk_play_out = net(talk_play_torch).detach().numpy()
    try: talk_play_out = nao_pca.inverse_transform(talk_play_out)
    except NameError: pass
    try: talk_play_out = nao_scaler.inverse_transform(talk_play_out)
    except NameError: pass
    # Before Smoothing
    print("Jerkiness before smoothing: {}".format(jerk(talk_play_out, 1.0 / h)))
    plot_joint_sequence([5, 20], nao_interface.joint_names, talk_play_out.T, nao_interface.limits, h)

    # Smoothing
    