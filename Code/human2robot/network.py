import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from itertools import permutations
from random import choice


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
        if self.upper:
            upper_index = x > self.upper
            x[upper_index] = self.upper[upper_index]

        if self.lower:
            lower_index = x < self.lower
            x[lower_index] = self.lower[lower_index]

        return x


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, 
                 AF='tanh', dropout_rate=0, learning_rate=0.1, max_epoch=2000,
                 joint_upper=None, joint_lower=None):
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
        self.cutAngle = CutAngle(joint_upper, joint_lower)
        self.AF = getActFunc(AF)
        self.dropout = nn.Dropout(dropout_rate)

        self.max_epoch = max_epoch
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        for layer in self.LayerList:
            x = self.AF(self.dropout(layer(x)))
        x = self.hidden2output(x)
        x = self.cutAngle(x)
        return x

    def __cross__(self, human, nao):
        pass

    def __train__(self, human_train, human_val, nao_train, nao_val, stop_rate=0.01):
        self.train_loss_list = []
        self.val_loss_list = []
        self.min_val_loss = np.inf
        for epoch in range(self.max_epoch):
            print('=====> Epoch: {}'.format(epoch))
            self.train()
            self.optimizer.zero_grad()
            train_loss = self.loss_func(self(human_train), nao_train)
            self.train_loss_list.append(train_loss.item())
            train_loss.backward()
            self.optimizer.step()

            self.eval()
            val_loss = self.loss_func(self(human_val), nao_val)
            self.val_loss_list.append(val_loss.item())
            if epoch > 25:
                if val_loss - self.min_val_loss > stop_rate * self.min_val_loss:
                    break
                elif val_loss < self.min_val_loss:
                    self.min_val_loss = val_loss
    
    def __test__(self, human_test, nao_test):
        self.eval()
        self.test_loss = self.loss_func(self(human_test), nao_test)

    def __plot__(self, save=False):
        plt.figure()
        plt.scatter(list(range(len(self.train_loss_list))), self.train_loss_list, s=1, c='blue')
        plt.scatter(list(range(len(self.val_loss_list))), self.val_loss_list, s=1, c='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.legend(['Training error', 'Validation error'])
        plt.show()
        if save:
            plt.savefig('result.png')

    @staticmethod
    def createFromRandomSearch(x_train, x_val, y_train, y_val, max_search=100):
        """Create a new Net based the hyper-parameters obtained from random search

        :param x_train: [description]
        :type x_train: torch.Tensor
        :param x_val: [description]
        :type x_val: torch.Tensor
        :param y_train: [description]
        :type y_train: torch.Tensor
        :param y_val: [description]
        :type y_val: torch.Tensor
        :param max_search: [description], defaults to 100
        :type max_search: int, optional
        """
        kwargs = Net.__randomsearch__(x_train, x_val, y_train, y_val, max_search)
        return Net(n_input=x_train.size(1), n_output=y_train.size(1), **kwargs)

    @staticmethod
    def __randomsearch__(x_train, x_val, y_train, y_val, max_search=100, filename=None):
        """Random search to find suitable hyper-parameter for the Network

        :param x_train: [description]
        :type x_train: torch.Tensor
        :param x_val: [description]
        :type x_val: torch.Tensor
        :param y_train: [description]
        :type y_train: torch.Tensor
        :param y_val: [description]
        :type y_val: torch.Tensor
        :param max_search: [description], defaults to 100
        :type max_search: int, optional
        """
        def generateHiddenLayerOptions(width_options=[32 * i for i in range(1,5)]):
            options = []
            for length in range(1, len(width_options)+1):
                options.extend(permutations(width_options, length))
            return list(map(list, options))

        af_options = ['relu', 'relu6', 'leaky_relu', 'celu', 'gelu', 'selu',
            'softplus', 'sigmoid', 'log_sigmoid', 'tanh']
        dr_options = [0.01 * i for i in range(50)]
        lr_options = [0.005 * (i + 1) for i in range(20)]
        hl_options = generateHiddenLayerOptions()
        options = [af_options, dr_options, lr_options, hl_options]
        keyword = ['AF', 'dropout_rate', 'learning_rate', 'n_hidden']

        if filename != None:
            file = open(filename, 'a')
        
        best_option = None
        best_val_error = np.inf
        for i in range(max_search):
            current_option = dict(zip(keyword, list(map(choice, options))))
            net = Net(n_input=x_train.size(1), n_output=y_train.size(1), **current_option)
            net.__train__(x_train, x_val, y_train, y_val)
            if filename != None:
                file.write(current_option['AF']+', ')
                file.write(str(current_option['dropout_rate'])+', ')
                file.write(str(current_option['learning_rate'])+', ')
                file.write(' '.join(list(map(str, current_option['n_hidden'])))+', ')
                file.write(str(float(net.min_val_loss))+'\n')
            if net.min_val_loss < best_val_error:
                best_val_error = float(net.min_val_loss)
                best_option = current_option

        print('Best hyper-parameter option: ', best_option)
        return best_option


def numpy2tensor(x):
    """Transter type numpy.ndarray to type torch.Tensor and change the precision to float
    Write in function so that can be used in functional programming
    
    :param x: The numpy data to be transformed
    :type x: numpy.ndarray
    :return: The torch data after transformation
    :rtype: torch.Tensor
    """
    return torch.from_numpy(x).float()