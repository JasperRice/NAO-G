from qpsolvers import solve_qp, solve_safer_qp
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import cvxopt
import math
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


def plot_joint_sequence(joints, jointNames, sequence, limits, h, title=None, start=None, end=None, filename=None):
    scale = 1.0 / 3.0
    for joint in joints:
        n = len(sequence[joint])
        angles = sequence[joint]
        vel = [(angles[i+1] - angles[i]) / h for i in range(n-1)]
        fig, axs = plt.subplots(2, sharex=True)

        ##### Joint Angle
        axs[0].plot(angles, '-', c='blue')
        axs[0].hlines(y=[limits['minAngle'][joint], limits['maxAngle'][joint]],
                      xmin=0, xmax=n-1, linestyles='dashed')
        width = limits['maxAngle'][joint] - limits['minAngle'][joint]

        if title: axs[0].set_title(title + " - Time Interval: %.4f s" % h, size=11)
        else: axs[0].set_title("Time Interval: %.4f s" % h, size=11)
        axs[0].set_ylabel('Joint angle / rad')
        axs[0].set_ylim([limits['minAngle'][joint]-scale*width, limits['maxAngle'][joint]+scale*width])
        axs[0].legend([jointNames[joint]], fontsize=8)

        ##### Joint Velocity
        axs[1].plot(vel, '-', c='blue')
        plt.hlines(y=[limits['maxChange'][joint], -limits['maxChange'][joint]],
                    xmin=0, xmax=n-2, linestyles='dashed')
        width = 2*limits['maxChange'][joint]

        axs[1].set_xlabel('Timestamp')
        axs[1].set_ylabel('Joint angular velocity / rad/s')
        axs[1].set_ylim([-limits['maxChange'][joint]-scale*width, limits['maxChange'][joint]+scale*width])
        axs[1].legend([jointNames[joint]], fontsize=8)
        
        #####
        if filename: plt.savefig('dataset/Figures/' + filename + '_Joint_' + jointNames[joint] + '.png')
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


    # Plot motion sequence on joints
    # human_interface.readJointAnglesFromBVH('dataset/BVH/NaturalTalking_030_2_1From5.bvh'); interval = 1.0 / 60.0 * 5
    # human_interface.readJointAnglesFromBVH('dataset/BVH/NaturalTalking_030_2.bvh'); interval = 1.0 / 60.0
    
    human_interface.readJointAnglesFromBVH('dataset/BVH/NaturalTalking_017_1From3.bvh'); interval = 1.0 / 60.0 * 3
    start = 3680; end = 3815
    # start = 5000; end = 5300
    # start = 8400; end = 8600

    human_sequence = human_interface.jointAngles[start:end]; human_sequence = np.delete(np.array(human_sequence), fingerIndex, axis=1)
    try: human_sequence = human_scaler.transform(human_sequence)
    except NameError: pass
    try: human_sequence = human_pca.transform(human_sequence)
    except NameError: pass

    human_sequence_torch = torch.from_numpy(human_sequence).float()
    nao_sequence = net(human_sequence_torch).detach().numpy()

    try: nao_sequence = nao_pca.inverse_transform(nao_sequence)
    except NameError: pass
    try: nao_sequence = nao_scaler.inverse_transform(nao_sequence)
    except NameError: pass

    # Before Post-processing
    print("Jerkiness before post-processing: {}".format(jerk(nao_sequence, 1.0/interval)))
    plot_joint_sequence([2,3,4,5,6,-5,-4,-3,-2,-1], nao_interface.joint_names, nao_sequence.T, nao_interface.limits, interval, title="Before Post-processing", filename="Before_Post-processing")
    # try: execGesture(P_NAO_IP, P_NAO_PORT, nao_sequence.tolist(), interval=interval, Interrupt=False)
    # except: execGesture(NAO_IP, NAO_PORT, nao_sequence.tolist(), interval=interval, Interrupt=False)

    # Test Control
    plotting = False
    smoothing = False
    constrained_optimization = True
    smoothing_after_optimization = True

    nao_sequence_smoothed = nao_sequence
    # Smoothing
    if smoothing:
        smooth_kwargs = {
            'window_length':    25, # 25 for 60FPS
            'polyorder':        3,
            'deriv':            0,
            'delta':            1.0,
            'axis':             -1,
            'mode':             'interp',
            'cval':             0.0
        }
        nao_sequence_smoothed = smooth(nao_sequence_smoothed, smoothing_method='savgol', **smooth_kwargs)
        print("Jerkiness after smoothing: {}".format(jerk(nao_sequence_smoothed, 1.0/interval)))
        if plotting:
            plot_joint_sequence([2,3,4,5,6,-5,-4,-3,-2,-1], nao_interface.joint_names, nao_sequence_smoothed.T, nao_interface.limits, interval, title="After Smoothing", filename="After_Smoothing")
        try: execGesture(P_NAO_IP, P_NAO_PORT, nao_sequence_smoothed.tolist(), interval=interval, Interrupt=False)
        except: execGesture(NAO_IP, NAO_PORT, nao_sequence_smoothed.tolist(), interval=interval, Interrupt=False)


    nao_sequence_smoothed_optimized = nao_sequence_smoothed
    # Constrained Optimization
    if constrained_optimization:
        all_limits = nao_interface.limits
        nao_sequence_smoothed_optimized = nao_sequence_smoothed_optimized.T

        use_qp = True
        if use_qp:
            n = nao_sequence_smoothed_optimized.shape[1]
            M1, M2 = np.eye(n, 2*n-1, k=0) * math.sqrt(5), np.eye(n-1, 2*n-1, k=n) * math.sqrt(0.01)
            M = np.vstack([M1, M2])
            P = np.dot(M.T, M)
            A = np.zeros_like(M2)
            for i in range(n-1): A[i, i], A[i, i+1], A[i, i+n] = -1, 1, -interval
            b = np.zeros(n-1).reshape((n-1,))
            for i, x in enumerate(nao_sequence_smoothed_optimized):
                limits = {'minAngle':     all_limits['minAngle'][i],
                          'maxAngle':     all_limits['maxAngle'][i],
                          'maxChange':    all_limits['maxChange'][i]}
                v = (x[1:] - x[:-1]) / interval
                s = np.hstack([x, v])
                s_origin = np.dot(M, s)
                q = np.negative(np.dot(M.T, s_origin))
                lb = np.array([limits['minAngle']]*n + [-limits['maxChange']]*(n-1))
                ub = np.array([limits['maxAngle']]*n + [limits['maxChange']]*(n-1))
                nao_sequence_smoothed_optimized[i] = solve_qp(P=P, q=q, A=A, b=b, lb=lb, ub=ub, solver='cvxopt')[:n]
        else:
            for i, x in enumerate(nao_sequence_smoothed_optimized):
                print("Optimizing joint {}:\t".format(i) + nao_interface.joint_names[i])
                x0 = x
                limits = {'minAngle':     all_limits['minAngle'][i],
                          'maxAngle':     all_limits['maxAngle'][i],
                          'maxChange':    all_limits['maxChange'][i]}
                cons = constraints(np.size(x), limits, interval)
                fun = cons[-1]['fun']
                args = (np.eye(np.size(x)) * 5.0, # 5.0
                        np.eye(np.size(x)-1) * 0.1, # 0.1
                        interval,
                        x
                    )
                sol = minimize(objective, x0, args=args, method='SLSQP', constraints=cons, options={'disp': False})
                nao_sequence_smoothed_optimized[i] = sol.x
        nao_sequence_smoothed_optimized = nao_sequence_smoothed_optimized.T
        print("Jerkiness after " + ("smoothing and optimization" if smoothing else "optimization") + ": {}".format(jerk(nao_sequence_smoothed_optimized, 1.0/interval)))
        if plotting:
            plot_joint_sequence([2,3,4,5,6,-5,-4,-3,-2,-1], nao_interface.joint_names, nao_sequence_smoothed_optimized.T, nao_interface.limits, interval, 
                                title="After Optimization" if not smoothing else "After Smoothing & Optimization",
                                filename="After_Optimization" if not smoothing else "After_Smoothing_Optimization")
        # try: execGesture(P_NAO_IP, P_NAO_PORT, nao_sequence_smoothed.tolist(), interval=interval, Interrupt=False)
        # except: execGesture(NAO_IP, NAO_PORT, nao_sequence_smoothed.tolist(), interval=interval, Interrupt=False)

    
    nao_sequence_optimized_smoothed = nao_sequence_smoothed_optimized
    if smoothing_after_optimization:
        smooth_kwargs = {
            'window_length':    9, # 25 for 60FPS; 9 For 20 FPS; 5 For 12 FPS
            'polyorder':        3,
            'deriv':            0,
            'delta':            1.0,
            'axis':             -1,
            'mode':             'interp',
            'cval':             0.0
        }
        nao_sequence_optimized_smoothed = smooth(nao_sequence_optimized_smoothed, smoothing_method='savgol', **smooth_kwargs)
        print("Jerkiness after " + ("optimization and smoothing" if constrained_optimization else "smoothing") + ": {}".format(jerk(nao_sequence_optimized_smoothed, 1.0/interval)))
        if plotting:
            plot_joint_sequence([2,3,4,5,6,-5,-4,-3,-2,-1], nao_interface.joint_names, nao_sequence_optimized_smoothed.T, nao_interface.limits, interval,
                                title="After Smoothing" if not constrained_optimization else "After Optimization & Smoothing",
                                filename="After_Smoothing" if not constrained_optimization else "After_Optimization_Smoothing")
        try: execGesture(P_NAO_IP, P_NAO_PORT, nao_sequence_optimized_smoothed.tolist(), interval=interval, Interrupt=False)
        except: execGesture(NAO_IP, NAO_PORT, nao_sequence_optimized_smoothed.tolist(), interval=interval, Interrupt=False)