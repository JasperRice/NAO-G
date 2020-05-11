from naoqi import ALProxy
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch

from io_routines import readCSV
from network import Net
from setting import *

class NAOInterface:
    
    def __init__(self, IP, PORT, JOINT_NANE='Joints', TIME_INTERVAL=0.5, **net_kwargs):
        self.motion = ALProxy("ALMotion", IP, PORT)
        self.joint_names = self.motion.getBodyNames(JOINT_NANE) # list
        # limits = np.array(self.motion.getLimits(JOINT_NANE)) # numpy.ndarray
        limits = torch.tensor(self.motion.getLimits(JOINT_NANE)) # torch.tensor
        self.limits = {
            'minAngle':     limits[:, 0],
            'maxAngle':     limits[:, 1],
            'maxChange':    limits[:, 2],
            'maxTorque':    limits[:, 3]
        }
        try: self.net = Net(**net_kwargs)
        except: pass

    def getJointNames(self):
        print('=====> Joint names:')
        print(self.joint_names)

    def getAngleLimits(self):
        print('=====> Lower bound:')
        print(self.limits['minAngle'])
        print('=====> Upper bound:')
        print(self.limits['maxAngle'])

    def loadNetwork(self, PATH):
        # self.net = xxx.load_state_dict(torch.load(PATH))
        pass

    def loadTalk(self, TALKFILE):
        self.talk = readCSV(TALKFILE)
        # self.joint_angles = self.net(self.talk)
        # self.cutAngles()

    def loadPoses(self, joint_angles):
        # joint_angles: torch.tensor / numpy.ndarray
        
        self.joint_angles = joint_angles.tolist()
        self.cutAngles()

        self.joint_name_angles = dict(zip(self.joint_names, self.joint_angles))

    def cutAngles(self):
        for joint in self.joint_angles:
            lower_index = joint < self.limits['minAngle']
            upper_index = joint > self.limits['maxAngle']
            joint[lower_index] = self.limits['minAngle'][lower_index]
            joint[upper_index] = self.limits['maxAngle'][upper_index]

    def executePoses(self):
        pass

    def generatePoses(self):
        pass

    def plotJoint(self, joint_name, *plt_args):
        try: plt.plot(self.joint_name_angles[joint_name], *plt_args)
        except ValueError: print('Warning: the joint name is not valid.')


if __name__ == "__main__":
    nao = NAOInterface(IP=NAO_IP, PORT=NAO_PORT)
    nao.getJointNames()
    nao.getAngleLimits()