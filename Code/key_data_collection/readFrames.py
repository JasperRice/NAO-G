import numpy as np


def read_motion(key_frames):
    key_motion = open("/home/jasper/Documents/NAO-G/src/data/NaturalTalking_002_KeyMotions.bvh", "a")
    key_motion.write("Frames: "+str(len(key_frames))+"\n")
    key_motion.write("Frame Time: 0.016667\n")

    whole_motion = open("data/NaturalTalking_002.bvh", "r")
    check = False
    for i, line in enumerate(whole_motion):
        if line == "MOTION\n":
            check = True
            key_frames += i
            key_frames += 2
        if check:
            if i in key_frames:
                key_motion.write(line)


if __name__ == "__main__":
    motion_001 = np.array([1, 48, 98, 187, 262, 290, 340, 690, 700, 710, 717, 786, 847, 1044, 1250, 1469, 1600, 1653, 1735, 1916, 2364, 2419, 2605, 2619, 2635, 2676, 2967, 3578, 3591, 3616, 3764, 4097, 4210, 5047, 5179, 5311, 5320, 5346, 5396, 5770, 5824, 6086, 6135, 6157, 6171, 6746, 7315, 7439, 7812, 8911, 8923, 9267, 9417, 9460, 9490, 9623, 9638, 9663, 9695, 9828, 9922, 9940, 9971, 10120, 10262, 10316, 10914, 10931, 10938, 10943, 10948, 10956, 11392, 11398, 11492, 11499, 11504, 11512, 11531, 11550, 11605, 11630, 11650, 11742, 12002, 12902])
    motion_002 = np.array([1, 24, 27, 35, 115, 200, 239, 334, 550, 594, 646, 740, 755, 1289, 1359, 1575, 1957, 2960, 3199, 3246, 6827, 7018, 9500, 9518, 9534, 9560, 9591, 9640, 9699, 11020, 11122, 11179, 11269])
    
    # key_frames = motion_001
    key_frames = motion_002
    read_motion(key_frames)