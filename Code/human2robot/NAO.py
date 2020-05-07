from naoqi import ALProxy
import numpy as np

class Interface:
    
    def __init__(self, IP, PORT, JOINT_NANE='Joints', TIME_INTERVAL=0.5):
        self.motion = ALProxy("ALMotion", IP, PORT)
        limits = np.array(self.motion.getLimits(JOINT_NANE))
        self.limits = {
            'minAngle':     limits[:, 0],
            'maxAngle':     limits[:, 1],
            'maxChange':    limits[:, 2],
            'maxTorque':    limits[:, 3]
        }

    def generatePoses(self):
        