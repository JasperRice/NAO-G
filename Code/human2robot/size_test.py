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

def choices(N, n):
    token = list(range(N))
    res = []
    for _ in range(n):
        ele = random.choice(token)
        token.remove(ele)
        res.append(ele)
    return res


human_interface = HumanInterface.createFromBVH('dataset/BVH/human_skeletion.bvh')
try: nao_interface = NAOInterface(IP=P_NAO_IP, PORT=P_NAO_PORT)
except:
    try: nao_interface = NAOInterface(IP=NAO_IP, PORT=NAO_PORT)
    except: pass

fingerIndex = []
fingerJointList = ['RightHandThumb1',   'RightHandThumb2',     'RightHandThumb3',
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

# Read dataset
talk_list = map(readCSV, talkfile)
talk = np.vstack(talk_list); talk = np.delete(talk, fingerIndex, axis=1)
talk, _ = normalize(talk); _, human_pca = decompose(talk)

human_data = readCSV('dataset/Human_overlap.csv'); human_data = np.delete(human_data, fingerIndex, axis=1)
human_test = readCSV('dataset/Human_test.csv'); human_test = np.delete(human_test, fingerIndex, axis=1)
nao_data = readCSV('dataset/NAO_overlap.csv'); nao_test = readCSV('dataset/NAO_test.csv')
n = np.size(human_data, 0)
if n != np.size(nao_data, 0):
    sys.exit("Numbers of input and target are different.")

results = {}
test_num = 20
file_val = open("size_test_val.csv", 'w')
file_test = open("size_test.csv", 'w')
for i in [25, 50, 75, 100, 125]:
    results[i] = {"val": [], "test": []}
    random.seed(1000)
    mask = choices(n, i)
    
    human = human_data[mask]
    human, human_scaler = normalize(human); human = human_pca.transform(human)
    __human_test__ = human_pca.transform(human_scaler.transform(human_test))

    nao = nao_data[mask]
    nao, nao_scaler = normalize(nao)
    __nao_test__ = nao_scaler.transform(nao_test)

    __human_test__ = torch.from_numpy(__human_test__).float()
    __nao_test__ = torch.from_numpy(__nao_test__).float()

    human_train, human_val, nao_train, nao_val = train_test_split(human, nao, test_size=0.2, random_state=500)
    human_train = torch.from_numpy(human_train).float()
    human_val = torch.from_numpy(human_val).float()
    nao_train = torch.from_numpy(nao_train).float()
    nao_val = torch.from_numpy(nao_val).float()

    net = Net(n_input=np.size(human, 1), n_hidden=N_HIDDEN, n_output=np.size(nao, 1),
              AF=AF, dropout_rate=0, learning_rate=L_RATE, max_epoch=1500)

    for j in range(test_num):
        net.__train__(human_train, human_val, nao_train, nao_val, max_epoch=3000, stop=True)
        net.__test__(__human_test__, __nao_test__)
        results[i]["val"].append(float(net.min_val_loss))
        results[i]["test"].append(float(net.test_loss))
    file_val.write(', '.join(map(str, results[i]["val"])) + '\n')
    file_test.write(', '.join(map(str, results[i]["test"])) + '\n')