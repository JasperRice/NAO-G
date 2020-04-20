import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    
    def __init__(self, n_input, n_hidden, n_output, AF='relu', dropout_rate=0):
        """[summary]
        
        :param n_input: [description]
        :type n_input: [type]
        :param n_hidden: The dimensions for each hidden layer.
        :type n_hidden: int
        :param n_output: [description]
        :type n_output: [type]
        """
        super(Net, self).__init__()
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


def numpy2tensor(x):
    """[summary]
    
    :param x: [description]
    :type x: [type]
    :return: [description]
    :rtype: [type]
    """    
    return torch.from_numpy(x).float()