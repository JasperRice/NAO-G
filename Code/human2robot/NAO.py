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
        self.joint_names = self.motion.getBodyNames(JOINT_NANE)
        limits = np.array(self.motion.getLimits(JOINT_NANE))
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
        self.joint_angles = joint_angles
        self.cutAngles()

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

    def plotJoint(self, joint_name):
        if joint_name in self.joint_names:
            pass


if __name__ == "__main__":
    nao = NAOInterface(IP=NAO_IP, PORT=NAO_PORT)
    nao.getJointNames()
    nao.getAngleLimits()