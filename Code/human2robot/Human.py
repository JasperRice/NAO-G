from copy import deepcopy
from torch import tensor
import pandas as pd

class HumanInterface:
    def __init__(self, jointNames, jointChannelCount, jointChannelNames):
        self.jointNames = jointNames
        self.jointChannelCount = jointChannelCount
        self.jointChannelNames = jointChannelNames
        self.jointAngles = []

    def __len__(self):
        return len(self.jointAngles)

    def __str__(self):
        return '\t'.join(self.jointNames)

    def readFromBVH(self, filenameList):
        for filename in filenameList:
            file = open(filename, 'r')
            lines = file.readlines()

            index = 0
            for i, line in enumerate(lines):
                wordList = line.split()
                if wordList[0] == 'MOTION':
                    index = i + 1
                    break
            self.jointInfo = lines[:index]
            index += 2
            while index < len(lines):
                self.jointAngles.append(list(map(float, lines[index].split())))
                index += 1
            file.close()
        self.jointAnglesBackup = deepcopy(self.jointAngles)

    def readFromCSV(self, filenameList):
        for filename in filenameList:
            df = pd.read_csv(filename)
            self.jointAngles += df.values.tolist()
        self.jointAnglesBackup = deepcopy(self.jointAngles)

    def writeToBVH(self, filename):
        file = open(filename, 'w')
        file.writelines(self.jointInfo)
        file.write('Frames: %d\n' % len(self))
        file.write('Frame Time: %f\n' % (1/60))
        for angle in self.jointAngles:
            file.write(' '.join(list(map(str, angle)))+'\n')
        file.close()

    def recoverAngles(self):
        self.jointAngles = deepcopy(self.jointAnglesBackup)

    def fixHips(self):
        for angle in self.jointAngles:
            angle[:3] = deepcopy([0]*3)

    def fixShoulders(self):
        indexR = (self.jointNames.index('RightShoulder')+1) * 3
        indexL = (self.jointNames.index('LeftShoulder')+1) * 3
        for angle in self.jointAngles:
            angle[indexR] = 90
            angle[indexL] = -90
            for i in [indexR+1, indexR+2, indexL+1, indexL+2]:
                angle[i] = 0

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
        return HumanInterface(*HumanInterface.readJointNames('Test.txt'))

    @staticmethod
    def readJointNames(filename):
        file = open(filename, 'r')
        lines = file.readlines()
        jointNames = []
        jointChannelCount = {}
        jointChannelNames = {}
        for index, line in enumerate(lines):
            wordList = list(line.split())
            if wordList[0] in ['ROOT', 'JOINT']:
                jointNames.append(wordList[1])
                jointChannelCount[wordList[1]] = 3
                jointChannelNames[wordList[1]] = []
            elif wordList[0] == 'CHANNELS':
                jointChannelCount[jointNames[-1]] = int(wordList[1])
                jointChannelNames[jointNames[-1]] = wordList[2:]
            elif wordList[0] == 'MOTION':
                break
        return jointNames, jointChannelCount, jointChannelNames


if __name__ == "__main__":
    human = HumanInterface.createFromBVH('Test.txt')
    human.readFromBVH([
        '/home/jasper/Documents/NAO-G/Code/key_data_collection/Talk_01_Key.bvh',
        '/home/jasper/Documents/NAO-G/Code/key_data_collection/Talk_02_Key.bvh',
        '/home/jasper/Documents/NAO-G/Code/key_data_collection/Talk_04_Key.bvh',
        '/home/jasper/Documents/NAO-G/Code/key_data_collection/Talk_05_Key.bvh'])
    human.fixHips()
    human.fixShoulders()
    human.writeToBVH('/home/jasper/Documents/NAO-G/Code/human2robot/write2BVH_Test.bvh')