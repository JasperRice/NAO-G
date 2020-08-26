"""
('Best hyper-parameter option: ', [0.0002296913506475621, 'relu', 0.005450020325607934, [128, 32]])
('Min validation loss: ', 0.5709372162818909)
"""
from itertools import permutations
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


def generateHiddenLayerOptions(width_options=[32 * i for i in range(1,5)]):
            options = []
            for length in range(1, len(width_options)+1):
                options.extend(permutations(width_options, length))
            return list(map(list, options))


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

human = readCSV('dataset/Human_overlap.csv'); human = np.delete(human, fingerIndex, axis=1)
human_test = readCSV('dataset/Human_test.csv'); human_test = np.delete(human_test, fingerIndex, axis=1)
nao = readCSV('dataset/NAO_overlap.csv')
nao_test = readCSV('dataset/NAO_test.csv')
n = np.size(human, 0)
if n != np.size(nao, 0):
    sys.exit("Numbers of input and target are different.")

human, human_scaler = normalize(human); human = human_pca.transform(human)
nao, nao_scaler = normalize(nao)

__human_test__ = human_pca.transform(human_scaler.transform(human_test))
__nao_test__ = nao_scaler.transform(nao_test)
__human_test__ = torch.from_numpy(__human_test__).float()
__nao_test__ = torch.from_numpy(__nao_test__).float()

human_train, human_val, nao_train, nao_val = train_test_split(human, nao, test_size=0.2, random_state=1000)
human_train = torch.from_numpy(human_train).float()
human_val = torch.from_numpy(human_val).float()
nao_train = torch.from_numpy(nao_train).float()
nao_val = torch.from_numpy(nao_val).float()


keywords = ['Learning Rate', 'Activation Function', 'Weight Decay', 'Hidden Layers', 'Dropout Rate','Validation Error']
lr_range = (-4, 0)
reg_range = (-5, -1)
hl_options = generateHiddenLayerOptions()
af_options = ['relu', 'leaky_relu', 'tanh', 'sigmoid']

try:
    file = open("random_search_result_with_dr_sigmoid_included.csv", 'r')
    lines = file.readlines()
    file.close()
    if len(lines) == 0:
        file = open("random_search_result_random_search_result_with_dr_sigmoid_includedwith_dr.csv", 'w')
        file.writelines(', '.join(keywords) + '\n')
    else:
        file = open("random_search_result_with_dr_sigmoid_included.csv", 'a')
except:
    file = open("random_search_result_with_dr_sigmoid_included.csv", 'w')
    file.writelines(', '.join(keywords) + '\n')

best_search = None
best_val_error = np.inf
num_search = 1000
for i in range(num_search):
    lr = 10 ** random.uniform(*lr_range)
    reg = 10 ** random.uniform(*reg_range)
    dr = random.uniform(0, 1)
    hidden_layers = random.choice(hl_options)
    af = random.choice(af_options)

    net = Net(n_input=np.size(human, 1), n_hidden=hidden_layers, n_output=np.size(nao, 1), 
              AF=af, dropout_rate=dr, learning_rate=lr, reg=reg)
    net.__train__(human_train, human_val, nao_train, nao_val, max_epoch=3000, stop=True)

    if net.min_val_loss < best_val_error:
        best_search = [lr, af, reg, hidden_layers, dr]
        best_val_error = float(net.min_val_loss)

    hidden_layers_str = ' '.join(map(str, hidden_layers))
    file.writelines(', '.join(map(str, [lr, af, reg, hidden_layers_str, dr, net.min_val_loss])) + '\n')

print('Best hyper-parameter option: ', best_search)
print('Min validation loss: ', best_val_error)