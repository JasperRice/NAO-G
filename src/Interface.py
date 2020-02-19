from bvh import Bvh
from naoqi import ALProxy

import argparse
import time


start = time.time()


def main(IP, PORT, FPS, FILE):
    """Interface of the motion retrageting
    
    :param IP: Adress IP of the NAO robot
    :type IP: str
    :param PORT: Port number of the NAO robot
    :type PORT: int
    :param FPS: [description]
    :type FPS: [type]
    :param FILE: [description]
    :type FILE: [type]
    """
    with open(FILE) as file:
        human_mocap = Bvh(file.read())
        # nframes = human_mocap.nframes
        # frame_time = human_mocap.frame_time
    
    nframes = 3
    frame_time = 1.0 / FPS
    
    motion = ALProxy("ALMotion", IP, PORT)
    memory = ALProxy("ALMemory", IP, PORT)

    motion.setAngles('LShoulderPitch', 0.0, 1.0)
    time.sleep(1.0)

    for index in range(nframes):

        motion.angleInterpolation('LShoulderPitch', 0.0, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.1, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.2, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.3, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.4, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.5, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.6, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.7, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.8, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.9, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 1.0, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.9, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.8, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.7, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.6, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.5, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.4, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.3, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.2, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.1, frame_time, True)
        motion.angleInterpolation('LShoulderPitch', 0.0, frame_time, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Address IP of the NAO robot.")
    parser.add_argument("--port", type=int, default=9559, help="Port number of the NAO robot.")
    parser.add_argument("--fps", type=int, default=20, help="FPS of motion of the NAO robot.")
    parser.add_argument("--filename", type=str, default="Gesture/Default.bvh", help="Filename of the human gesture.")
    args = parser.parse_args()
    main(args.ip, args.port, args.fps, args.filename)
