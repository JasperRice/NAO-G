import numpy as np


class MotionTrajectorySynchronization:

    def __init__(self, nopt_seq):
        self.nopt_seq = nopt_seq.copy()
        self.opt_seq = np.zeros_like(self.nopt_seq)
        
        self.P = np.size(self.nopt_seq, axis=1)
        
        self.Q1 = 5 * np.eye(self.P)
        self.Q2 = 0.1 * np.eye(self.P)
        self.Q3 = 10 * np.eye(self.P)

        self.domain = [0.5, 1]

    def getRobotConstraints(self):
        pass
        
    def getRobotDynamics(self):
        pass

    def optimizeSequence(self):
        pass

    def constrainedOptimization(self):
        pass

    def piecewiseOptimization(self):
        pass

    def jointConstraintViolated(self) -> bool:
        return True

    def notLargeAmplitudeMotion(self) -> bool:
        return True