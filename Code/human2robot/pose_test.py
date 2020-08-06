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

    # Read traing dataset
    human = readCSV('dataset/Human_overlap.csv'); human = np.delete(human, fingerIndex, axis=1)
    nao = readCSV('dataset/NAO_overlap.csv')
    if np.size(human, 0) != np.size(nao, 0): sys.exit("Numbers of input and target are different.")

    # Normalize and decompose the dataset
    human, human_scaler = normalize(human)
    nao, nao_scaler = normalize(nao)

    # Load model
    net = torch.load("net.txt")
    net.eval()
    
    # Load human poses
    human_pose_test = readCSV('dataset/Human_Type_Hands_Mean.csv'); human_pose_test = np.delete(human_pose_test, fingerIndex, axis=1)
    human_pose_test = human_scaler.transform(human_pose_test)
    human_pose_test_torch = torch.from_numpy(human_pose_test).float()

    # Forward pass human poses to the model
    nao_pose_test_torch = net(human_pose_test_torch)
    nao_pose_test = nao_pose_test_torch.detach().numpy()
    try: nao_pose_test = nao_pca.inverse_transform(nao_pose_test)
    except NameError: pass
    try: nao_pose_test = nao_scaler.inverse_transform(nao_pose_test)
    except NameError: pass
    raw_input("Press ENTER to execute the test results.")
    try: execGesture(P_NAO_IP, P_NAO_PORT, nao_pose_test.tolist())
    except: execGesture(NAO_IP, NAO_PORT, nao_pose_test.tolist())