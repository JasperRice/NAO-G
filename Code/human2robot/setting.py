NAO_IP      = "127.0.0.1"   # The IP address of the NAO robot, could be virtual or physical robot
NAO_PORT    = 41133         # The port number of the NAO robot

VISUALIZE   = True          # If visualize the training result on NAO based on the IP and Port defined
ON_SET      = 1             # Visualize on [0: train, 1: validation or 2: test]
PLAY_TALK   = not VISUALIZE # If play the sequence
PLAY_SET    = 0

USE_TALK    = True          # If use the whole Natural Talking dataset to decompose
USE_HAND    = False         # If use the hand data recorded in the dataset
NORMALIZE   = True          # If normalize dataset
DECOMPOSE   = False         # If use PCA to decompose dataset

MAX_EPOCH   = 10000         # The maximum training epoch
AF          = 'leaky_relu'  # Activation function ['leaky_relu', 'relu', 'sigmoid', 'tanh']
N_HIDDEN    = 128           # The number of nodes in the hidden layer
L_RATE      = 0.1           # The learning rate of the network
DO_RATE     = 0.25          # Dropout rate of the hidden layer
STOP_EARLY  = True          # If stop earlier based on the validation error
STOP_THRES  = 0.005         # If the current validation error is STOP_THRES larger than the lowest error, stop training
STOP_RATE   = 0.05

SAVE_DATA   = False         # If save the shuffled human gesture dataset
PATHNAME    = "human2robot/dataset/"
FILENAME    = ["TALK_01.csv", "TALK_02.csv", #"TALK_04.csv", "TALK_05.csv",
                "HUMAN.csv",
                "NAO.csv"]
filename    = map(lambda x: PATHNAME + x, FILENAME)