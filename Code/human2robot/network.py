import torch
import torch.nn as nn
import torch.nn.functional as F


def getActFunc(AF='tanh'):
    actFuncDict = nn.ModuleDict({
        'relu':             nn.ReLU(),
        'relu6':            nn.ReLU6(),
        'leaky_relu':       nn.LeakyReLU(),
        'celu':             nn.CELU(),
        'gelu':             nn.GELU(),
        'selu':             nn.SELU(),
        'softplus':         nn.Softplus(),
        'sigmoid':          nn.Sigmoid(),
        'log_sigmoid':      nn.LogSigmoid(),
        'tanh':             nn.Tanh()
    })
    if AF not in actFuncDict:
        print('Warning: Activation function is invalid. Using tanh instead.')
        AF = 'tanh'
    return actFuncDict[AF]


class CutAngle(nn.Module):
    def __init__(self, upper, lower):
        """A layer that cut the input based on upper and lower bound

        :param upper: [description]
        :type upper: [type]
        :param lower: [description]
        :type lower: [type]
        """
        super(CutAngle, self).__init__()
        self.upper = upper
        self.lower = lower

    def forward(self, x):
        upper_index = x > self.upper
        lower_index = x < self.lower
        x[upper_index] = self.upper[upper_index]
        x[lower_index] = self.lower[lower_index]
        return x


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, 
                 AF='tanh', dropout_rate=0,
                 joint_upper=None, joint_lower=None, 
                 **af_kwargs):
        """The feed forward neural network with multiple hidden layers
        
        :param n_input: The dimension of the input layer
        :type n_input: int
        :param n_hidden: The dimensions of the hidden layers
        :type n_hidden: list[int]
        :param n_output: The dimension of the output layer
        :type n_output: int
        :param AF: The activation function to be used, defaults to 'tanh'
        :type AF: str, optional
        :param dropout_rate: The dropout rate of the hidden layer, defaults to 0
        :type dropout_rate: int, optional
        :param joint_upper: The upper bound of allowed joints of NAO (after Normalization)
        :type joint_upper: np.ndarray
        :param joint_lower: The lower bound if allowed joints of NAO (after Normalization)
        :type joint_lower: np.ndarray
        """
        super(Net, self).__init__()

        # Define each layer here:
        self.LayerList = nn.ModuleList([nn.Linear(n_input, n_hidden[0])])
        self.LayerList.extend(nn.Linear(n_hidden[i], n_hidden[i+1]) for i in range(len(n_hidden)-1))
        self.hidden2output = nn.Linear(n_hidden[-1], n_output)
        # self.cutAngle = CutAngle(joint_upper, joint_lower)
        self.AF = getActFunc(AF)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for layer in self.LayerList:
            x = self.AF(self.dropout(layer(x)))
        x = self.hidden2output(x)
        # x = self.cutAngle(x)
        return x

    def __train__(self):
        pass


def numpy2tensor(x):
    """Transter type numpy.ndarray to type torch.Tensor and change the precision to float
    Write in function so that can be used in functional programming
    
    :param x: The numpy data to be transformed
    :type x: numpy.ndarray
    :return: The torch data after transformation
    :rtype: torch.Tensor
    """
    return torch.from_numpy(x).float()