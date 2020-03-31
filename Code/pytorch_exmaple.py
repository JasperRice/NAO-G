import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

class Net(nn.Module):
    
    def __init__(self, n_input, n_hiddens, n_output):
        """[summary]
        
        :param n_input: [description]
        :type n_input: [type]
        :param n_hiddens: A list of integers that define the dimensions for each hidden layer.
        :type n_hiddens: list
        :param n_output: [description]
        :type n_output: [type]
        """
        super(Net, self).__init__()

        # Define each layer here:
        self.input2hidden = nn.Linear(n_input, n_hiddens[0])

        self.hidden2hidden = []
        for i in range(len(n_hiddens)-1):
            self.hidden2hidden.append(nn.Linear(n_hiddens[i], n_hiddens[i+1]))

        self.hidden2output = nn.Linear(n_hiddens[-1], n_output)


    def forward(self, x):
        x = F.relu(self.input2hidden(x))
        
        for h2h in self.hidden2hidden:
            x = h2h(x)

        x = self.hidden2output(x)
        
        return x

net = Net(n_input=30, n_hiddens=[40, 50, 100], n_output=25)
print(net)

# Normalization