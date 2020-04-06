import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    
    def __init__(self, n_input, n_hidden, n_output):
        """[summary]
        
        :param n_input: [description]
        :type n_input: [type]
        :param n_hidden: The dimensions for each hidden layer.
        :type n_hidden: int
        :param n_output: [description]
        :type n_output: [type]
        """
        super(Net, self).__init__()

        # Define each layer here:
        self.input2hidden = nn.Linear(n_input, n_hidden)
        self.hidden2output = nn.Linear(n_hidden, n_output)


    def forward(self, x):
        x = F.relu(self.input2hidden(x))
        x = self.hidden2output(x)
        return x


if __name__ == "__main__":
    net = Net(n_input=30, n_hidden=40, n_output=25)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = nn.MSELoss()
    