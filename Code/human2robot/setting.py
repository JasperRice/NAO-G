NAO_IP      = '127.0.0.1'   # The IP address of the NAO robot, could be virtual or physical robot
NAO_PORT    = 36399         # The port number of the NAO robot

P_NAO_IP      = '130.237.67.140'
P_NAO_PORT    = 9559

VISUALIZE   = False         # If visualize the training result on NAO based on the IP and Port defined
ON_SET      = 1             # Visualize on [0: train, 1: validation or 2: test]
PLAY_TALK   = not VISUALIZE # If play the sequence
PLAY_SET    = 0             # The index of the talk

USE_TALK    = True          # If use the whole Natural Talking dataset to decompose
USE_HAND    = False         # If use the hand data recorded in the dataset
NORMALIZE   = True          # If normalize dataset
DECOMPOSE   = False         # If use PCA to decompose dataset

MAX_EPOCH   = 1000          # The maximum training epoch
AF          = 'tanh'        # Activation function ['leaky_relu', 'relu', 'sigmoid', 'tanh']
N_HIDDEN    = [128, 64]     # The number of nodes in the hidden layers
L_RATE      = 0.1           # The learning rate of the network
DO_RATE     = 0.25          # Dropout rate of the hidden layer
STOP_EARLY  = True          # If stop earlier based on the validation error
STOP_RATE   = 0.01          # If the validation error is this ratio higher than the minimum error, the training would stop

SAVE_DATA   = False         # If save the shuffled human gesture dataset
PATHNAME    = "/home/nao/Documents/NAO-G/Code/human2robot/dataset/"
FILENAME    = ["TALK_01.csv", "TALK_02.csv", #"TALK_04.csv", "TALK_05.csv",
                "Human.csv",
                "NAO.csv"]
filename    = map(lambda x: PATHNAME + x, FILENAME)