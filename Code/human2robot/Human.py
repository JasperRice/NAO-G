from copy import deepcopy
from torch import tensor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

    def normalize(self, dataset=None):
        if dataset == None:
            self.human_scaler = StandardScaler.fit(np.array(self.jointAngles))
        else:
            self.human_scaler = StandardScaler.fit(dataset)

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
    # print(human)

    # human.readFromCSV('/home/nao/Documents/NAO-G/Code/human2robot/dataset/Human_extra.csv')
    # human.readFromCSV('/home/nao/Documents/NAO-G/Code/human2robot/dataset/Human.csv')
    # human.fixHips()
    # human.fixShoulders()
    # human.writeToBVH('/home/nao/Documents/NAO-G/Code/human2robot/test_bvh.bvh')
    
    # human.readFromBVH([
    #     '/home/jasper/Documents/NAO-G/Code/key_data_collection/Talk_01_Key.bvh',
    #     '/home/jasper/Documents/NAO-G/Code/key_data_collection/Talk_02_Key.bvh',
    #     '/home/jasper/Documents/NAO-G/Code/key_data_collection/Talk_04_Key.bvh',
    #     '/home/jasper/Documents/NAO-G/Code/key_data_collection/Talk_05_Key.bvh'])

    human.readJointAnglesFromBVH('/home/nao/Documents/NAO-G/Code/key_data_collection/NaturalTalking_001.bvh')
    # human.fixFingers()
    # human.fixHips()
    # human.fixLegs()
    human.fixShoulders()
    # human.fixSpines()
    human.writeJointAnglesToBVH('/home/nao/Documents/NAO-G/Code/human2robot/NaturalTalking_001_FixedShoulder.bvh')
    # print(len(human))