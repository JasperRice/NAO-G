import numpy as np
from scipy.optimize import minimize


class constrained_optimization:
    
    def __init__(self, nao_interface, nao_motion):
        self.limits = nao_interface.limits
        self.motion = nao_motion

def linearQuaratic(x_opt, *x):
    pass


if __name__ == "__main__":
    def calcVolume(x):
        length = x[0]
        width = x[1]
        height = x[2]
        volume = length * width * height
        return volume

    def calcSurface(x):
        length = x[0]
        width = x[1]
        height = x[2]
        surface = 2 * (length*width + width*height + height*length)
        return surface

    def objective(x):
        return -calcVolume(x)

    def constraint(x):
        return 10 - calcSurface(x)

    def length_constraint(x):
        return 1 - x[0]

    cons = ({
        'type': 'ineq',
        'fun': constraint
    })

    len_cons = ({
        'type': 'ineq',
        'fun': lambda x: 1 - x[0]
    })

    # Initial guess
    lengthGuess = 1
    widthGuess = 1
    heightGuess = 1

    x0 = np.array([lengthGuess, widthGuess, heightGuess])

    sol = minimize(objective, x0, method='SLSQP', constraints=[cons, len_cons], options={'disp': True})
    print(sol.x)