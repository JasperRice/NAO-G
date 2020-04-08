from naoqi import ALProxy
import argparse
import time

start = time.time()

def execute(IP, PORT, FPS, FILE):
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
    motion = ALProxy("ALMotion", IP, PORT)

    G = [-0.1503739356994629, -0.37433791160583496, 1.6873581409454346, 0.01683211326599121, -1.5877318382263184, -1.3084601163864136, -0.8314700126647949, -0.40646815299987793, 0.047595977783203125, 0.3083760738372803, 0.04904603958129883, -0.042994022369384766, -0.022968053817749023, -0.40646815299987793, -0.17176604270935059, 0.2269899845123291, -0.06285214424133301, 0.13196587562561035, 0.1104898452758789, 1.5723919868469238, -0.013848066329956055, 1.9680800437927246, 0.4019498825073242, -0.49245595932006836]
    T = [0.5] * len(G)
    motion.angleInterpolation("Joints", G, T, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Address IP of the NAO robot.")
    parser.add_argument("--port", type=int, default=34975, help="Port number of the NAO robot.")
    parser.add_argument("--fps", type=int, default=30, help="FPS of motion of the NAO robot.")
    parser.add_argument("--filename", type=str, default="Gesture/Default.bvh", help="Filename of the human gesture.")
    args = parser.parse_args()

    execute(args.ip, args.port, args.fps, args.filename)