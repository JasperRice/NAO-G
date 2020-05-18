from .Human import HumanInterface
from .NAO import NAOInterface
from .network import Net

class Human2NAOInterface:

    def __init__(self, human, nao, net):
        self.human = human
        self.nao = nao
        self.net = net