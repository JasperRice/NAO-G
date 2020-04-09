# from bvh import Bvh
from naoqi import ALProxy

import argparse
import time


start = time.time()


def saveJointAngles(IP, PORT, FILENAME, HUMANMOTION, LHAND, RHAND):
    motion = ALProxy("ALMotion", IP, PORT)
    # robot_joint_names = motion.getBodyNames("Joints")
    """[
        'HeadYaw', 'HeadPitch',
        'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw',
        'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
        'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll',
        'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw'
    ]"""

    f = open(FILENAME, "a")
    f.write(HUMANMOTION + "\t: " + str(LHAND) + ", " + str(RHAND) + ", " + str(motion.getAngles("Joints", True)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="130.237.67.140", help="Address IP of the NAO robot.")
    parser.add_argument("--port", type=int, default=9559, help="Port number of the NAO robot.")
    parser.add_argument("--fps", type=int, default=30, help="FPS of motion of the NAO robot.")
    parser.add_argument("--filename", type=str, default="Gesture/Default.bvh", help="Filename of the human gesture.")
    args = parser.parse_args()

    saveJointAngles(args.ip, args.port, "./data.txt", "T4F1", True, True)
    # main(args.ip, args.port, args.fps, args.filename)