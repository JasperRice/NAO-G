import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, AF='relu', dropout_rate=0):
        """The feed forward neural network with one single hidden layer
        
        :param n_input: The dimension of the input layer
        :type n_input: int
        :param n_hidden: The dimension of the hidden layer
        :type n_hidden: int
        :param n_output: The dimension of the output layer
        :type n_output: int
        :param AF: The activation function to be used, defaults to 'relu'
        :type AF: str, optional
        :param dropout_rate: The dropout rate of the hidden layer, defaults to 0
        :type dropout_rate: int, optional
        """   
        super(Net, self).__init__()

        self.activations = nn.ModuleDict({
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

        self.AF = AF

        # Define each layer here:
        self.input2hidden = nn.Linear(n_input, n_hidden)
        self.hidden2output = nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.AF == 'leaky_relu':
            x = F.leaky_relu(self.dropout(self.input2hidden(x)))
        elif self.AF == 'relu':
            x = F.relu(self.dropout(self.input2hidden(x)))
        elif self.AF == 'sigmoid':
            x = F.sigmoid(self.dropout(self.input2hidden(x)))
        elif self.AF == 'tanh':
            x = F.tanh(self.dropout(self.input2hidden(x)))
        else:
            print('Warning: Activation function is invalid. Use Relu instead.')
            x = F.relu(self.dropout(self.input2hidden(x)))
        x = self.hidden2output(x)
        return x


class MultiLayerNet(nn.Module):
    def __init__(self, n_input, n_hiddens, n_output, AF='relu', dropout_rate=0):
        """The feed forward neural network with multiple hidden layers
        TODO: Add threshold layer at the end of the net
        
        :param n_input: The dimension of the input layer
        :type n_input: int
        :param n_hidden: The dimensions of the hidden layers
        :type n_hidden: list[int]
        :param n_output: The dimension of the output layer
        :type n_output: int
        :param AF: The activation function to be used, defaults to 'relu'
        :type AF: str, optional
        :param dropout_rate: The dropout rate of the hidden layer, defaults to 0
        :type dropout_rate: int, optional
        """   
        super(MultiLayerNet, self).__init__()

        self.activations = nn.ModuleDict({
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

        if AF not in self.activations:
            print('Warning: Activation function is invalid. Using Relu instead.')
            AF = 'relu'

        # Define each layer here:
        self.LayerList = nn.ModuleList([nn.Linear(n_input, n_hiddens[0])])
        self.LayerList.extend(nn.Linear(n_hiddens[i], n_hiddens[i+1]) for i in range(len(n_hiddens)-1))
        # self.LayerList.extend(nn.Linear(n_hiddens[-1], n_output))
        self.hidden2output = nn.Linear(n_hiddens[-1], n_output)
        self.AF = self.activations[AF]
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for layer in self.LayerList:
            x = self.AF(self.dropout(layer(x)))
        x = self.hidden2output(x)
        return x


class CutAngleLayer(nn.Module):
    def __init__(self, upper, lower):
        """[summary]

        :param nn: [description]
        :type nn: [type]
        :param upper: [description]
        :type upper: [type]
        :param lower: [description]
        :type lower: [type]
        """        
        super(CutAngleLayer, self).__init__()
        self.upper = upper
        self.lower = lower

    def forward(self, x):
        return x


def numpy2tensor(x):
    """Transter type numpy.ndarray to type torch.Tensor and change the precision to float
    Write in function so that can be used in functional programming
    
    :param x: The numpy data to be transformed
    :type x: numpy.ndarray
    :return: The torch data after transformation
    :rtype: torch.Tensor
    """    
    return torch.from_numpy(x).float()