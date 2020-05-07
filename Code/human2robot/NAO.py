from naoqi import ALProxy
from torch import nn
import numpy as np
import torch

class Interface:
    
    def __init__(self, IP, PORT, JOINT_NANE='Joints', TIME_INTERVAL=0.5, **net_kwargs):
        self.motion = ALProxy("ALMotion", IP, PORT)
        limits = np.array(self.motion.getLimits(JOINT_NANE))
        self.limits = {
            'minAngle':     limits[:, 0],
            'maxAngle':     limits[:, 1],
            'maxChange':    limits[:, 2],
            'maxTorque':    limits[:, 3]
        }
        # self.net = Net(**net_kwargs)

    def loadNetwork(self, PATH):
        # self.net = xxx.load_state_dict(torch.load(PATH))
        pass

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