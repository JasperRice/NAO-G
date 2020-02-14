from naoqi import ALProxy

import argparse
import time


start = time.time()


def main(IP, PORT, FPS):
    motion  = ALProxy("ALMotion", IP, PORT)
    motion.post.setAngles(['LShoulderRoll', 'LShoulderPitch', 'LElbowRoll', 'LElbowYaw'],
                          [0.0, 0.0, 0.0, 0.0],
                          1.0)
    time.sleep(1.0 / FPS)

    motion.post.setAngles(['LShoulderRoll', 'LShoulderPitch', 'LElbowRoll', 'LElbowYaw'],
                          [0.0, 0.5, 0.0, 0.0],
                          1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Address IP of the NAO robot.")
    parser.add_argument("--port", type=int, default=9559, help="Port number of the NAO robot.")
    parser.add_argument("--fps", type=int, default=30, help="FPS of the gesture.")
    args = parser.parse_args()
    main(args.ip, args.port, args.fps)
