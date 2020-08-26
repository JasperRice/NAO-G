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


def plot_joint_sequence(joints, jointNames, sequence, limits, h, filename=None):
    for joint in joints:
        n = len(sequence[joint])
        angles = sequence[joint]
        vel = [(angles[i+1] - angles[i]) / h for i in range(n-1)]

        plt.plot(angles, '-', c='blue')
        plt.hlines(y=[limits['minAngle'][joint], limits['maxAngle'][joint]],
                    xmin=0, xmax=n-1, linestyles='dashed')
        plt.xlabel('Timestamp')
        plt.ylabel('Joint angle / rad')
        plt.legend([jointNames[joint]])
        plt.show()

        plt.plot(vel, '-', c='blue')
        plt.hlines(y=[limits['maxChange'][joint], -limits['maxChange'][joint]],
                    xmin=0, xmax=n-2, linestyles='dashed')
        plt.xlabel('Timestamp')
        plt.ylabel('Joint angular velocity / rad/s')
        plt.legend([jointNames[joint]])
        plt.show()


def jerk(all_motion, f):
    joints_sequence = all_motion.T
    J = 0.0
    for S in joints_sequence:
        N = len(S)
        for i in range(N-3):
            J += ((S[i+3] - 3.0*S[i+2] + 3.0*S[i+1] - S[i]) * (f**3))**2
    
    return J / (2*f*(N-3))


if __name__ == "__main__":
    # Reproducibility by setting a seed
    torch.manual_seed(2020)

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

    # Read dataset
    talk_list = map(readCSV, talkfile)
    talk = np.vstack(talk_list); talk = np.delete(talk, fingerIndex, axis=1)
    
    # human = readCSV('dataset/Human.csv'); human = np.delete(human, fingerIndex, axis=1)
    # human_new = readCSV('dataset/Human_new.csv'); human_new = np.delete(human_new, fingerIndex, axis=1)
    # human_right_hand = readCSV('dataset/Human_right_hand.csv'); human_right_hand = np.delete(human_right_hand, fingerIndex, axis=1)
    # human_new_agu = human_new + np.random.normal(loc=0.0, scale=0.6, size=np.shape(human_new))
    # human_right_hand_agu = human_right_hand + np.random.normal(loc=0.0, scale=0.6, size=np.shape(human_right_hand))
    human_overlap = readCSV('dataset/Human_overlap.csv'); human_overlap = np.delete(human_overlap, fingerIndex, axis=1)
    human = np.vstack([
        # human,
        # human_new,
        # human_new_agu,
        # human_right_hand,
        # human_right_hand_agu,
        human_overlap
    ])
    
    # nao = readCSV('dataset/NAO.csv')
    # nao_new = readCSV('dataset/NAO_new.csv')
    # nao_new_agu = nao_new + np.random.normal(loc=0.0, scale=0.009, size=np.shape(nao_new))
    # nao_right_hand = readCSV('dataset/NAO_right_hand.csv')
    # nao_right_hand_agu = nao_right_hand + np.random.normal(loc=0.0, scale=0.009, size=np.shape(nao_right_hand))
    nao_overlap = readCSV('dataset/NAO_overlap.csv')
    nao = np.vstack([
        # nao,
        # nao_new,
        # nao_new_agu,
        # nao_right_hand,
        # nao_right_hand_agu,
        nao_overlap
    ])

    n = np.size(human, 0)
    if n != np.size(nao, 0):
        sys.exit("Numbers of input and target are different.")

    # Normalize and decompose the dataset
    talk, _ = normalize(talk)
    _, human_pca = decompose(talk)
    human, human_scaler = normalize(human)
    # human, human_pca = decompose(human)
    human = human_pca.transform(human)
    nao, nao_scaler = normalize(nao)

    # Split the dataset into train, test, and validation
    # dataset = split(human, nao)
    human_train, human_val, nao_train, nao_val = train_test_split(human, nao, test_size=0.2, random_state=1000)
    human_train_torch   = torch.from_numpy(human_train).float()
    human_val_torch     = torch.from_numpy(human_val).float()
    nao_train_torch     = torch.from_numpy(nao_train).float()
    nao_val_torch       = torch.from_numpy(nao_val).float()

    # Save the shuffled train and validation data and visualize the ground truth of the NAO gestures
    SAVE_DATA = False
    if SAVE_DATA:
        human_to_save = [human_train, human_val]
        nao_to_save = [nao_train, nao_val]
        try:
            human_to_save = map(human_pca.inverse_transform, human_to_save)
            nao_to_save = map(nao_pca.inverse_transform, nao_to_save)
        except NameError: pass
        try:
            human_to_save = map(human_scaler.inverse_transform, human_to_save)
            nao_to_save = map(nao_scaler.inverse_transform, nao_to_save)
        except NameError: pass
        np.savetxt("human_train.txt", human_to_save[0])
        np.savetxt("human_val.txt", human_to_save[1])
        # np.savetxt("human_test.txt", human_to_save[2])
        np.savetxt("nao_train.txt", nao_to_save[0])
        np.savetxt("nao_val.txt", nao_to_save[1])
        # np.savetxt("nao_test.txt", nao_to_save[2])
        execGesture(NAO_IP, NAO_PORT, nao_to_save[ON_SET][:,2:].tolist()) \
            if USE_HAND else execGesture(NAO_IP, NAO_PORT, nao_to_save[ON_SET].tolist())
        exit()

    # Define Neural Network and train
    net = Net(
        n_input=np.size(human, 1),
        n_hidden=N_HIDDEN,
        n_output=np.size(nao, 1),
        AF=AF, dropout_rate=DO_RATE, learning_rate=L_RATE, reg=REG, ues_lr_scheduler=False
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
        # human_interface.readJointAnglesFromBVH('dataset/BVH/NaturalTalking_030_2_1From5.bvh'); h = 1.0 / 60.0 * 5.0
        human_interface.readJointAnglesFromBVH('dataset/BVH/NaturalTalking_030_2.bvh'); h = 1.0 / 60.0
        talk_play = human_interface.jointAngles[:500]                
        talk_play = np.array(talk_play)
        talk_play = np.delete(talk_play, fingerIndex, axis=1)
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


    # =====> Plot before smoothing
    print("Jerkiness before smoothing: {}".format(jerk(nao_out, 1.0 / h)))
    plot_joint_sequence([5, 20], nao_interface.joint_names, nao_out.T, nao_interface.limits, h)


    smooth_kwargs = {
        'window_length':    7,
        'polyorder':        3,
        'deriv':            0,
        'delta':            1.0,
        'axis':             -1,
        'mode':             'interp',
        'cval':             0.0
    }
    # =====> Smoothing
    nao_out = smooth(nao_out, smoothing_method='savgol', **smooth_kwargs)
    # =====> Plot after smoothing
    print("Jerkiness after smoothing: {}".format(jerk(nao_out, 1.0 / h)))
    plot_joint_sequence([5, 20], nao_interface.joint_names, nao_out.T, nao_interface.limits, h)


    all_limits = nao_interface.limits
    nao_out = nao_out.T
    # =====> Constrained Optimization
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
        sol = minimize(objective, x0, args=args, method='SLSQP', constraints=cons, options={'disp': False})
        nao_out[i] = sol.x
    # =====> Piecewise Constrained Optimization
    # for i, x in enumerate(nao_out):
    #     if i not in (list(range(2,7))+list(range(19,24))):
    #         continue
    #     print("Optimizing joint {}.".format(i))
    #     r = 20
    #     x_a_0 = np.hstack([x, np.ones(r)])
    #     limits = {
    #         'minAngle':     all_limits['minAngle'][i],
    #         'maxAngle':     all_limits['maxAngle'][i],
    #         'maxChange':    all_limits['maxChange'][i],
    #         'maxA':         1,
    #         'minA':         0.5,
    #     }
    #     args = (np.eye(np.size(x)) * 5.0,
    #             np.eye(np.size(x)-1) * 0.1,
    #             np.eye(r) * 10,
    #             h,
    #             x,
    #             r
    #            )
    #     cons = piecewise_constraints(x_a_0, limits, h, r)
    #     sol = minimize(piecewise_objective, x_a_0, args=args, method='SLSQP', constraints=cons, options={'disp': True})
    #     nao_out[i] = (sol.x)[:-r]
    nao_out = nao_out.T


    # =====> Plot after constrained optimization
    print("Jerkiness after constrained optimization: {}".format(jerk(nao_out, 1.0 / h)))
    plot_joint_sequence([5, 20], nao_interface.joint_names, nao_out.T, nao_interface.limits, h, filename="cons_opt")


    # =====> Execute the motion
    raw_input("Press ENTER to execute the motion.")
    try: execGesture(P_NAO_IP, P_NAO_PORT, nao_out.tolist(), TIME=h)
    except: execGesture(NAO_IP, NAO_PORT, nao_out.tolist(), TIME=h)