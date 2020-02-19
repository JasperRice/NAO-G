from naoqi import ALProxy

import argparse
import bvh
import time


start = time.time()


def main(IP, PORT, FPS):
    """Interface of the motion retrageting
    
    :param IP: Adress IP of the NAO robot
    :type IP: str
    :param PORT: Port number of the NAO robot
    :type PORT: int
    """
    motion  = ALProxy("ALMotion", IP, PORT)
    # memory  = ALProxy("ALMemory", IP, PORT)

    # for _ in range(50):
        # motion.post.setAngles(['LShoulderRoll', 'LShoulderPitch', 'LElbowRoll', 'LElbowYaw'],
        #                     [0.0, 0.0, 0.0, 0.5],
        #                     1.0)
        # time.sleep(1.0/5.0)

        # motion.post.setAngles(['LShoulderRoll', 'LShoulderPitch', 'LElbowRoll', 'LElbowYaw'],
        #                     [0.0, 0.5, 0.5, -0.5],
        #                     1.0)
        # time.sleep(1.0/5.0)

    motion.angleInterpolation('LShoulderPitch', 0.0, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.1, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.2, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.3, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.4, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.5, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.6, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.7, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.8, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.9, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 1.0, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.9, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.8, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.7, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.6, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.5, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.4, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.3, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.2, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.1, 1.0/20, True)
    motion.angleInterpolation('LShoulderPitch', 0.0, 1.0/20, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Address IP of the NAO robot.")
    parser.add_argument("--port", type=int, default=9559, help="Port number of the NAO robot.")
    parser.add_argument("--fps", type=int, default=30, help="FPS of motion of the NAO robot.")
    parser.add_argument("--filename", type=str, default="", help="FPS of motion of the NAO robot.")
    args = parser.parse_args()
    main(args.ip, args.port, args.fps)
