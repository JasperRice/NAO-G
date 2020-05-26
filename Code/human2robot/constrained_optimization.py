import numpy as np
from scipy.optimize import minimize


def constraints(x_opt, limits):
    cons = []
    min_x = limits['minAngle']
    max_x = limits['maxAngle']
    min_v = -limits['maxChange']
    max_v = limits['maxChange']
    cons.append({'type': 'ineq', 'fun': lambda x_opt: max_x - x_opt[0]})
    cons.append({'type': 'ineq', 'fun': lambda x_opt: x_opt[0] - min_x})
    for i in range(1, np.size(x_opt)):
        cons.append({'type': 'ineq', 'fun': lambda x_opt: max_x - x_opt[i]})
        cons.append({'type': 'ineq', 'fun': lambda x_opt: x_opt[i] - min_x})
        cons.append({'type': 'ineq', 'fun': lambda x_opt: max_v - (x_opt[i] - x_opt[i-1])})
        cons.append({'type': 'ineq', 'fun': lambda x_opt: (x_opt[i] - x_opt[i-1]) - min_v})

    return cons


def objective(x_opt, **kwargs):
    q1 = kwargs['q1']
    q2 = kwargs['q2']
    x = kwargs['x']
    v = x[1:] - x[:-1]
    v_opt = x_opt[1:] - x_opt[:-2]

    x_delta = np.atleast_2d(x_opt - x)
    v_delta = np.atleast_2d(v_opt - v)
    return float(np.matmul(np.matmul(x_delta, q1), x_delta.T) + np.matmul(np.matmul(v_delta, q2), v_delta.T))


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
        'fun': lambda x: - 0.3 + (x[0]-x[1])
    })

    len_cons_2 = ({
        'type': 'ineq',
        'fun': lambda x: 1 - x[0]
    })

    # Initial guess
    lengthGuess = 1
    widthGuess = 1
    heightGuess = 1

    x0 = np.array([lengthGuess, widthGuess, heightGuess])

    sol = minimize(objective, x0, method='SLSQP', constraints=[cons, len_cons, len_cons_2], options={'disp': True})
    print(sol.x)