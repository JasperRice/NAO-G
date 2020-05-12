class HumanInterface:
    
    def __init__(self, jointNames, jointChannelCount, jointChannelNames):
        self.jointNames = jointNames
        self.jointChannelCount = jointChannelCount
        self.jointChannelNames = jointChannelNames

    @staticmethod
    def createFromFile(filename):
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


    def fixJoint(self):
        pass


human = HumanInterface.createFromFile('Test.txt')
print(human.jointNames)