from naoqi import ALProxy
import argparse
import time

start = time.time()

def execGesture(IP, PORT, GESTURE, TIME=None):
    """Execute a static gesture on virtual or physical NAO robot
    
    :param IP: Adress IP of the NAO robot
    :type IP: str
    :param PORT: Port number of the NAO robot
    :type PORT: int
    :param GESTURE: A list of lists of joint angles of NAO
    :type GESTURE: list[list]
    :param TIME: A list of lists of time
    :type TIME: list[list]
    """
    motion = ALProxy("ALMotion", IP, PORT)
    if TIME == None:
        TIME = [[0.5] * len(GESTURE[0]) for _ in range(len(GESTURE))]
    motion.angleInterpolation("Joints", GESTURE, TIME, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Address IP of the NAO robot.")
    parser.add_argument("--port", type=int, default=34975, help="Port number of the NAO robot.")
    args = parser.parse_args()

    gesture = [0.09660005569458008, -0.10895586013793945, 1.6674160957336426, 0.2730100154876709, -1.1766200065612793, -0.28835010528564453, -0.08594608306884766, -0.16869807243347168, 0.31297802925109863, 0.11662602424621582, -0.09208202362060547, 0.1288139820098877, -0.2269899845123291, -0.16869807243347168, 0.08287787437438965, 0.14722204208374023, -0.08739614486694336, 0.09668397903442383, -0.0014920234680175781, 1.5785279273986816, -0.15804386138916016, 1.2271580696105957, 0.27002596855163574, 0.13955211639404297]
    execGesture(args.ip, args.port, gesture)