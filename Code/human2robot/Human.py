class HumanInterface:
    
    def __init__(self):
        pass

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

        return jointNames, jointChannelCount, jointChannelNames


jointList, channelCount, channelNames = HumanInterface.readJointNames('Test.txt')
jointName = jointList[1]
print(jointName)
print(channelCount[jointName])
print(channelNames[jointName])