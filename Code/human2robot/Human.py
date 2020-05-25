from copy import deepcopy
from math import radians
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import tensor

import pandas as pd
import numpy as np


class HumanInterface:
    def __init__(self, jointNames, jointStartIndex, lenJointAngles, jointChannelCount, jointChannelNames, jointInfo):
        self.jointNames = jointNames
        self.jointNamesBackup = deepcopy(self.jointNames)
        self.jointStartIndex = jointStartIndex
        self.jointStartIndexBackup = deepcopy(self.jointStartIndex)
        self.lenJointAngles = lenJointAngles
        self.jointChannelCount = jointChannelCount
        self.jointChannelNames = jointChannelNames
        self.jointInfo = jointInfo
        self.jointAngles = []
        self.jointAnglesBackup = []

    def __len__(self):
        return self.lenJointAngles

    def __str__(self):
        return ' '.join(self.jointNames)

    def __iter__(self):
        for joint in self.jointAngles:
            yield joint

    def __getitem__(self, i):
        return self.jointAngles[i]

    def normalize_fit(self, dataset=None):
        if dataset == None:
            self.human_scaler = StandardScaler.fit(np.array(self.jointAngles))
        else:
            self.human_scaler = StandardScaler.fit(dataset)

    def normalize_transform(self):
        self.jointAngles = self.human_scaler.transform(self.jointAngles)

    def normalize_inverse(self):
        self.jointAngles = self.human_scaler.inverse_transform(self.jointAngles)

    def printAngleLabels(self):
        angleLabels = []
        for joint in self.jointNames:
            for channel in self.jointChannelNames[joint]:
                angleLabels.append(joint+'_'+channel)

        print(', '.join(angleLabels))

    def recoverAngles(self):
        self.jointNames = deepcopy(self.jointNamesBackup)
        self.jointStartIndex = deepcopy(self.jointStartIndexBackup)
        self.jointAngles = deepcopy(self.jointAnglesBackup)
    
    def getStartAngleIndex(self, jointName):
        return self.jointStartIndex[self.jointNames.index(jointName)]

    def generateJointAngles(self, jointNameList=['RightArm','RightForeArm','RightHand','LeftArm','LeftForeArm','LeftHand']):
        if jointNameList is str:
            jointNameList = [jointNameList]
        for jointName in jointNameList:
            pass

    def addJointAngles(self, jointAngleDict):
        for joint in jointAngleDict:
            pass

    def readJointAnglesFromBVH(self, filenameList):
        self.jointAngles = []
        self.addJointAnglesFromBVH(filenameList)

    def readJointAnglesFromCSV(self, filenameList):
        self.jointAngles = []
        self.addJointAnglesFromCSV(filenameList)

    def addJointAnglesFromBVH(self, filenameList):
        if type(filenameList) is str:
            filenameList = [filenameList]
        for filename in filenameList:
            file = open(filename, 'r')
            lines = file.readlines()
            index = 0
            for i, line in enumerate(lines):
                wordList = line.split()
                if wordList[0] == 'MOTION':
                    index = i + 3
                    break
            while index < len(lines):
                self.jointAngles.append(list(map(float, lines[index].split())))
                index += 1
            file.close()
        self.jointAnglesBackup = deepcopy(self.jointAngles)

    def addJointAnglesFromCSV(self, filenameList):
        if type(filenameList) is str:
            filenameList = [filenameList]
        for filename in filenameList:
            df = pd.read_csv(filename)
            self.jointAngles += df.values.tolist()
        self.jointAnglesBackup = deepcopy(self.jointAngles)

    def writeJointAnglesToBVH(self, filename):
        file = open(filename, 'w')
        file.writelines(self.jointInfo)
        file.write('Frames: %d\n' % len(self.jointAngles))
        file.write('Frame Time: %f\n' % (1.0/60.0))
        for angle in self.jointAngles:
            file.write(' '.join(list(map(str, angle)))+'\n')
        file.close()

    def writeJointAnglesToCSV(self, filename):
        file = open(filename, 'w')
        file.write(', '.join(self.jointNames)+'\n')
        

    def fixFingers(self):
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
        for joint in fingerJointList:
            index = self.getStartAngleIndex(joint)
            for angle in self.jointAngles:
                angle[index:index+3] = deepcopy([0]*3)

    def fixLegs(self):
        legJointList = [
            'RightUpLeg',       'RightLeg',     'RightFoot',
            'RightForeFoot',    'RightToeBase',
            'LeftUpLeg',        'LeftLeg',      'LeftFoot',
            'LeftForeFoot',     'LeftToeBase'
        ]
        for joint in legJointList:
            index = self.getStartAngleIndex(joint)
            for angle in self.jointAngles:
                angle[3:6] = deepcopy([0]*3)
                angle[index:index+3] = deepcopy([0]*3)

    def fixHips(self):
        for angle in self.jointAngles:
            angle[:3] = deepcopy([0]*3)

    def fixShoulders(self):
        indexR = self.getStartAngleIndex('RightShoulder')
        indexL = self.getStartAngleIndex('LeftShoulder')
        for angle in self.jointAngles:
            angle[indexR] = 90
            angle[indexL] = -90
            for i in [indexR+1, indexR+2, indexL+1, indexL+2]:
                angle[i] = 0

    def fixSpines(self):
        spineJointList = ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head']
        for joint in spineJointList:
            index = self.getStartAngleIndex(joint)
            for angle in self.jointAngles:
                angle[index:index+self.jointChannelCount[joint]] = deepcopy([0]*self.jointChannelCount[joint])

    def head(self):
        for i in range(5):
            print('=====> Frame %d:' % i)
            print(' '.join(list(map(str, self.jointAngles[i]))))

    def tail(self):
        for i in range(-6, 0):
            print('=====> Frame %d:' % (len(self)+i))
            print(' '.join(list(map(str, self.jointAngles[i]))))

    def transformJointAnglesToNAO(self, nao, jointAngle):
        naoJointAngle = [0] * len(nao)
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
            naoIndex = nao.getJointIndex(naoJoint)
            humanIndex = self.getStartAngleIndex(humanJoint[0]) + humanJoint[1]
            if naoJoint in ['RShoulderRoll', 'LElbowYaw']:
                naoJointAngle[naoIndex] = radians(jointAngle[humanIndex] - 90)
            elif naoJoint in ['LShoulderRoll', 'RElbowYaw']:
                naoJointAngle[naoIndex] = radians(jointAngle[humanIndex] + 90)
            elif naoJoint in ['RShoulderPitch', 'LShoulderPitch']:
                naoJointAngle[naoIndex] = radians(- jointAngle[humanIndex] + 90)
            elif naoJoint in ['RWristYaw', 'RElbowRoll']:
                naoJointAngle[naoIndex] = radians(jointAngle[humanIndex])
            elif naoJoint in ['LWristYaw']:
                naoJointAngle[naoIndex] = radians(jointAngle[humanIndex] - 180)
            elif naoJoint in ['LElbowRoll']:
                naoJointAngle[naoIndex] = radians(- jointAngle[humanIndex])
        return naoJointAngle

    @staticmethod
    def createFromBVH(filename):
        return HumanInterface(*HumanInterface.readJointNames(filename))

    @staticmethod
    def readJointNames(filename):
        file = open(filename, 'r')
        lines = file.readlines()
        jointNames = []
        jointStartIndex = [0]
        jointChannelCount = {}
        jointChannelNames = {}
        index = 0
        for i, line in enumerate(lines):
            wordList = list(line.split())
            if wordList[0] in ['ROOT', 'JOINT']:
                jointNames.append(wordList[1])
                jointStartIndex.append(0)
                jointChannelCount[wordList[1]] = 3
                jointChannelNames[wordList[1]] = []
            elif wordList[0] == 'CHANNELS':
                jointStartIndex[-1] = jointStartIndex[-2] + int(wordList[1])
                jointChannelCount[jointNames[-1]] = int(wordList[1])
                jointChannelNames[jointNames[-1]] = wordList[2:]
            elif wordList[0] == 'MOTION':
                index = i + 1
                break
        lenJointAngles =  jointStartIndex.pop()
        jointInfo = lines[:index]
        return jointNames, jointStartIndex, lenJointAngles, jointChannelCount, jointChannelNames, jointInfo


if __name__ == "__main__":
    human = HumanInterface.createFromBVH('/home/nao/Documents/NAO-G/Code/human2robot/human_skeletion.bvh')
    human.readJointAnglesFromCSV('/home/nao/Documents/NAO-G/Code/human2robot/dataset/human_right_hand.csv')
    human.fixShoulders()

    human.writeJointAnglesToBVH('/home/nao/Documents/NAO-G/Code/human2robot/dataset/human_right_hand.bvh')