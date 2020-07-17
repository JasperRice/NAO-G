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
                   'LeftHandPinky1',    'LeftHandPinky2',      'LeftHandPinky3']
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
if np.size(human, 0) != np.size(nao, 0): sys.exit("Numbers of input and target are different.")

human, human_scaler = normalize(human)
nao, nao_scaler = normalize(nao)

__human_test__ = human_scaler.transform(human_test)
__human_test__ = torch.from_numpy(__human_test__).float()


num_try = 100
results = {}
ablation = 'reg'
file_train = open('ablations_'+ablation+'_results_train.csv', 'w')
file_val = open('ablations_'+ablation+'_results_val.csv', 'w')
file_test = open('ablations_'+ablation+'_results_test.csv', 'w')
options = [0, 0.005450020325607934]
# options = np.logspace(-5, -1, num=20, endpoint=False)
for opt in options:
    results[opt] = {'train':[], 'val': [], 'test': []}
    for i in range(num_try):
        human_train, human_val, nao_train, nao_val = train_test_split(human, nao, test_size=0.2)
        human_train = torch.from_numpy(human_train).float()
        human_val = torch.from_numpy(human_val).float()
        nao_train = torch.from_numpy(nao_train).float()
        nao_val = torch.from_numpy(nao_val).float()

        net = Net(n_input=np.size(human, 1), n_hidden=[128, 32], n_output=np.size(nao, 1),
                  AF='relu', dropout_rate=0.0765078199812, learning_rate=0.0002296913506475621, reg=opt)
        net.__train__(human_train, human_val, nao_train, nao_val, max_epoch=5000, stop=True, stop_rate=0.01)

        net.eval()
        nao_test_result = net(__human_test__).detach().numpy()
        try: nao_test_result = nao_scaler.inverse_transform(nao_test_result)
        except: pass
        test_loss = net.loss_func(nao_test_result, nao_test).item()
        
        print(float(net.min_val_loss), float(test_loss))
        results[opt]['train'].append(float(net.train_loss_list[-1]))
        results[opt]['val'].append(float(net.min_val_loss))
        results[opt]['test'].append(float(test_loss))
    file_train.writelines(', '.join(map(str, results[opt]['train'])) + '\n')
    file_val.writelines(', '.join(map(str, results[opt]['val'])) + '\n')
    file_test.writelines(', '.join(map(str, results[opt]['test'])) + '\n')

# x_ticks_labels = dp_options
fig, ax = plt.subplots(1,1)
for mode in ['val', 'test']:
    y = []
    e = []
    for dp in options:
        res_arr = np.array(results[dp][mode])
        y.append(np.mean(res_arr))
        e.append(np.std(res_arr))

    x = options
    y = np.array(y)
    e = np.array(e)

    ax.errorbar(x, y, e, linestyle='--', marker='.', fmt='-o', label=mode)
    # ax.set_xticks(x) # Set number of ticks for x-axis
    # ax.set_xticklabels(x_ticks_labels) # Set ticks labels for x-axis rotation='vertical'

plt.xscale('log')
plt.xlabel("Learning rate")
plt.ylabel("Error with standard deviation")
plt.legend()
plt.show()