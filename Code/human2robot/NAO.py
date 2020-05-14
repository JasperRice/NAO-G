from naoqi import ALProxy
from torch import nn
import matplotlib.pyplot as plt
import math
import numpy as np
import torch

from io_routines import readCSV
from network import Net
from setting import *

class NAOInterface:
    
    def __init__(self, IP, PORT, JOINT_NANE='Joints', TIME_INTERVAL=0.5, **net_kwargs):
        self.jointSetName = JOINT_NANE
        self.motion = ALProxy("ALMotion", IP, PORT)
        self.joint_names = self.motion.getBodyNames(self.jointSetName) # list
        # limits = np.array(self.motion.getLimits(self.jointSetName)) # numpy.ndarray
        limits = torch.tensor(self.motion.getLimits(self.jointSetName)) # torch.tensor
        self.limits = {
            'minAngle':     limits[:, 0],
            'maxAngle':     limits[:, 1],
            'maxChange':    limits[:, 2],
            'maxTorque':    limits[:, 3]
        }
        self.jointAngles = []
        try: self.net = Net(**net_kwargs)
        except: pass

    def readFromCSV(self, filenameList):
        if filenameList is str:
            filenameList = [filenameList]
        for filename in filenameList:
            pass

    def getJointAngleDegrees(self):
        for joint in self.jointAngles:
            print(np.degrees(joint))

    def getJointNames(self):
        print('=====> Joint names:')
        print(self.joint_names)

    def getAngleLimits(self):
        print('=====> Lower bound:')
        print(self.limits['minAngle'])
        print('=====> Upper bound:')
        print(self.limits['maxAngle'])

    def addNewJointAngles(self):
        self.jointAngles.append(self.motion.getAngles(self.jointSetName, True))

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

    def stand(self):
        J = [0.0, 0.0, 1.5476394891738892, 0.1388729065656662, -1.5734877586364746, -0.044828496873378754, 2.802596928649634e-45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5476394891738892, -0.1388729065656662, 1.5734877586364746, 0.044828496873378754, -2.802596928649634e-45]
        T = [0.5] * len(J)
        self.motion.angleInterpolation("Joints", J, T, True)


if __name__ == "__main__":
    nao = NAOInterface(IP=NAO_IP, PORT=NAO_PORT)
    # nao.stand()

    nao.addNewJointAngles()
    print(nao.jointAngles)
    nao.getJointAngleDegrees()
