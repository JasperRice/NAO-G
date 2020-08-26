NAO_IP      = '127.0.0.1'   # The IP address of the NAO robot, could be virtual or physical robot
NAO_PORT    = 35009         # The port number of the NAO robot

P_NAO_IP    = '130.237.67.8'
P_NAO_PORT  = 9559

VISUALIZE   = False         # If visualize the training result on NAO based on the IP and Port defined
ON_SET      = 1             # Visualize on [0: train or 1: validation]
PLAY_TALK   = not VISUALIZE # If play the sequence
PLAY_SET    = 0             # The index of the talk

USE_HAND    = False         # If use the hand data recorded in the dataset
USE_TALK    = True          # If use the whole Natural Talking dataset to decompose human poses
NORMALIZE   = True          # If normalize dataset
DECOMPOSE   = False         # If use PCA to decompose dataset

MAX_EPOCH   = 5000          # The maximum training epoch
AF          = 'relu'        # Activation function ['leaky_relu', 'relu', 'sigmoid', 'tanh']
N_HIDDEN    = [128, 64]     # The number of nodes in the hidden layers
L_RATE      = 0.131718496669    # The learning rate of the network
REG         = 0.0093777674026   # The weight decay 
DO_RATE     = 0.21          # Dropout rate of the hidden layer
# STOP_EARLY  = True          # If stop earlier based on the validation error
# STOP_RATE   = 0.01          # If the validation error is this ratio higher than the minimum error, the training would stop

SAVE_DATA   = False         # If save the shuffled human gesture dataset
PATHNAME    = "dataset/"
FILENAME    = ["NaturalTalking_001.csv", 
               "NaturalTalking_002.csv",
               "NaturalTalking_004.csv",
               "NaturalTalking_005.csv"]
talkfile    = map(lambda x: PATHNAME + x, FILENAME)