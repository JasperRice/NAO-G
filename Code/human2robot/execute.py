from naoqi import ALProxy
import argparse
import numpy as np
import time

start = time.time()


def readJointLimit(IP, PORT):
    """Read the joint limits

    :param IP: Adress IP of the NAO robot
    :type IP: str
    :param PORT: Port number of the NAO robot
    :type PORT: int
    :return: Limits of the joints: minAngle, maxAnlge, maxVelocity, maxTorque
    :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """
    motion = ALProxy("ALMotion", IP, PORT)
    limits = np.array(motion.getLimits("Joints"))
    return limits[:, 0], limits[:, 1], limits[:, 2], limits[:,3]


def cutJointAngles(IP, PORT, JOINT):
    JOINT = np.array(JOINT)
    lower, upper, _, _ = readJointLimit(IP, PORT)

    for joint in JOINT:
        lower_index = joint < lower
        upper_index = joint > upper
        joint[lower_index] = lower[lower_index]
        joint[upper_index] = upper[upper_index]

    return JOINT.tolist()


def execGesture(IP, PORT, JOINT, TIME=None, Interrupt=True):
    """Execute a static gesture on virtual or physical NAO robot
    
    :param IP: Adress IP of the NAO robot
    :type IP: str
    :param PORT: Port number of the NAO robot
    :type PORT: int
    :param JOINT: A list of lists of joint angles of NAO
    :type JOINT: list[list]
    :param TIME: A list of lists of time
    :type TIME: list[list]
    """
    motion = ALProxy("ALMotion", IP, PORT)
    JOINT = cutJointAngles(IP, PORT, JOINT)
    if TIME is None: TIME = [[1.0/60*20] * len(JOINT[0]) for _ in range(len(JOINT))]
    else: TIME = [[TIME] * len(JOINT[0]) for _ in range(len(JOINT))]

    motion.angleInterpolation("Joints", JOINT[0], 0.5, True)
    raw_input("Press ENTER to start ...")
    i = 0
    for J, T in zip(JOINT, TIME):
        i += 1
        print("=====> Motion index: %d" % i)
        motion.angleInterpolation("Joints", J, T, True)
        if Interrupt: raw_input("\tPress ENTER to continue ...")


if __name__ == "__main__":
    lower, upper, _, _ = readJointLimit("127.0.0.1", 36877)
    print lower