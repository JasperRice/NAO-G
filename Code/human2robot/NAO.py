from copy import deepcopy
from math import degrees
from naoqi import ALProxy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from Human import HumanInterface
from io_routines import readCSV
from network import Net
from setting import *

class NAOInterface:
    
    def __init__(self, IP, PORT, JOINT_NANE='Joints', TIME_INTERVAL=0.5):
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
        self.jointAnglesBackup = []

    def __str__(self):
        return ' '.join(self.joint_names)
    
    def __iter__(self):
        for joint in self.jointAngles:
            yield joint

    def __getitem__(self, i):
        return self.jointAngles[i]

    def readFromCSV(self, filenameList):
        self.jointAngles = []
        if type(filenameList) is str:
            filenameList = [filenameList]
        for filename in filenameList:
            df = pd.read_csv(filename)
            self.jointAngles += df.values.tolist()
        self.jointAnglesBackup = deepcopy(self.jointAngles)

    def writeToCSV(self, filename, mode='a'):
        file = open(filename, mode=mode)
        if mode == 'w':
            file.write(', '.join(self.joint_names)+'\n') 
            for angle in self.jointAngles:
                file.write(', '.join(list(map(str, angle)))+'\n')
        elif mode == 'a':
            for angle in self.jointAngles:
                file.write(', '.join(list(map(str, angle)))+'\n')
        file.close()

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

    def getJointIndex(self, jointName):
        return self.joint_names.index(jointName)

    def getCurrentJointAngle(self):
        return self.motion.getAngles(self.jointSetName, True)

    def addCurrentJointAngleToList(self):
        self.jointAngles.append(self.getCurrentJointAngle())

    def startRecordJointAngles(self):
        self.jointAngles = []
        CONTINUE = True
        while CONTINUE:
            nao.addCurrentJointAngleToList()
            if_continue = raw_input("If continue to record current joint angle (y/n): ")
            if if_continue in ['y', 'Y', 'yes']:
                CONTINUE = True
            elif if_continue in ['n', 'N', 'no']:
                CONTINUE = False
            else:
                if_continue = raw_input("Input 'y' or 'n': ")

    def transformJointAnglesListToHuman(self, human):
        for jointAngle in self.jointAngles:
            humanJointAngle = self.transformJointAnglesToHuman(human, jointAngle)
            human.jointAngles.append(humanJointAngle)

    def transformJointAnglesToHuman(self, human, jointAngle):
        humanJointAngle = [0] * len(human)
        humanJointAngle[human.getStartAngleIndex('RightShoulder')] = 90
        humanJointAngle[human.getStartAngleIndex('LeftShoulder')] = -90
        naoJointList = [
            'RShoulderRoll', 'LShoulderRoll', 'RShoulderPitch', 'LShoulderPitch',
            'RElbowRoll', 'LElbowRoll', 'RElbowYaw', 'LElbowYaw',
            'RWristYaw', 'LWristYaw'
        ]
        humanJointList = [
            ['RightArm', 0], ['LeftArm', 0], ['RightArm', 1], ['LeftArm', 1],
            ['RightForeArm', 1], ['LeftForeArm', 1], ['RightArm', 2], ['LeftArm', 2],
            ['RightForeArm', 2], ['LeftForeArm', 2]
        ]
        for naoJoint, humanJoint in zip(naoJointList, humanJointList):
            naoIndex = self.getJointIndex(naoJoint)
            humanIndex = human.getStartAngleIndex(humanJoint[0]) + humanJoint[1]
            if naoJoint in ['RShoulderRoll', 'LElbowYaw']:
                humanJointAngle[humanIndex] = degrees(jointAngle[naoIndex]) + 90
            elif naoJoint in ['LShoulderRoll', 'RElbowYaw']:
                humanJointAngle[humanIndex] = degrees(jointAngle[naoIndex]) - 90
            elif naoJoint in ['RShoulderPitch', 'LShoulderPitch']:
                humanJointAngle[humanIndex] = - degrees(jointAngle[naoIndex]) + 90
            elif naoJoint in ['RWristYaw', 'LWristYaw', 'RElbowRoll']:
                humanJointAngle[humanIndex] = degrees(jointAngle[naoIndex])
            elif naoJoint in ['LElbowRoll']:
                humanJointAngle[humanIndex] = - degrees(jointAngle[naoIndex])
            # print(naoJoint, naoIndex, degrees(jointAngle[naoIndex]), humanJoint, humanIndex, humanJointAngle[humanIndex])
        return humanJointAngle

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

    def executePosesOneByOne(self, index=None):
        T = [0.5] * len(self.joint_names)
        if index:
            print("=====> Pose index: %d." % (index))
            self.motion.angleInterpolation(self.jointSetName, self.joint_names[index], T, True)
        else:
            for index, angle in enumerate(self.jointAngles):
                print("=====> Pose index: %d." % (index))
                self.motion.angleInterpolation(self.jointSetName, angle, T, True)
                raw_input("Press ENTER to continue ...")

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
    human = HumanInterface.createFromBVH('/home/nao/Documents/NAO-G/Code/human2robot/human_skeletion.bvh')
    nao = NAOInterface(IP=NAO_IP, PORT=NAO_PORT)
    print(nao)
    nao.stand()

    # CONTINUE = True
    # while CONTINUE:
    #     nao.addCurrentJointAngleToList()
    #     if_continue = raw_input("If continue to record current joint angle (y/n):")
    #     if if_continue in ['y', 'Y', 'yes']:
    #         CONTINUE = True
    #     elif if_continue in ['n', 'N', 'no']:
    #         CONTINUE = False
    #     else:
    #         if_continue = raw_input("Input 'y' or 'n': ", end='')
    # nao.writeToCSV('/home/nao/Documents/NAO-G/Code/human2robot/dataset/NAO_extra.csv', mode='a')
    # nao.transformJointAnglesListToHuman(human)
    # human.writeToBVH('/home/nao/Documents/NAO-G/Code/human2robot/Human_extra.bvh')

    nao.readFromCSV('/home/nao/Documents/NAO-G/Code/human2robot/dataset/NAO_extra.csv')
    nao.transformJointAnglesListToHuman(human)
    human.writeJointAnglesToBVH('/home/nao/Documents/NAO-G/Code/human2robot/Human_extra.bvh')
    nao.executePosesOneByOne()