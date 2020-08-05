import numpy as np
from scipy.optimize import minimize


def constraints(size, limits, h):
    cons = []
    min_x = limits['minAngle']
    max_x = limits['maxAngle']
    min_v = -limits['maxChange']
    max_v = limits['maxChange']
    cons.append({'type': 'ineq', 'fun': lambda x: np.array(max_x - x[0])})
    cons.append({'type': 'ineq', 'fun': lambda x: np.array(x[0] - min_x)})
    for i in range(1, size):
        cons.append({'type': 'ineq', 'fun': lambda x: np.array(max_x - x[i])})
        cons.append({'type': 'ineq', 'fun': lambda x: np.array(x[i] - min_x)})
        cons.append({'type': 'ineq', 'fun': lambda x: np.array(max_v - (x[i] - x[i-1]) / h)})
        cons.append({'type': 'ineq', 'fun': lambda x: np.array((x[i] - x[i-1]) / h - min_v)})

    return cons

def objective(x_opt, *args):
    q1 = args[0]
    q2 = args[1]
    h = args[2]
    x = args[3]
    v = (x[1:] - x[:-1]) / h
    v_opt = (x_opt[1:] - x_opt[:-1]) / h

    x_delta = np.atleast_2d(x_opt - x)
    v_delta = np.atleast_2d(v_opt - v)
    return float(np.matmul(np.matmul(x_delta, q1), x_delta.T) + np.matmul(np.matmul(v_delta, q2), v_delta.T))


def piecewise_constraints(x_a, limits, h, r):
    x = x_a[:-r]
    a = x_a[-r:]
    cons = []
    min_x = limits['minAngle']
    max_x = limits['maxAngle']
    min_v = -limits['maxChange']
    max_v = limits['maxChange']
    max_a = 1
    min_a = 0.5
    cons.append({'type': 'ineq', 'fun': lambda x: max_x - x[0]})
    cons.append({'type': 'ineq', 'fun': lambda x: x[0] - min_x})
    for i in range(1, np.size(x)):
        cons.append({'type': 'ineq', 'fun': lambda x: max_x - x[i]})
        cons.append({'type': 'ineq', 'fun': lambda x: x[i] - min_x})
        cons.append({'type': 'ineq', 'fun': lambda x: max_v - (x[i] - x[i-1]) / h})
        cons.append({'type': 'ineq', 'fun': lambda x: (x[i] - x[i-1]) / h - min_v})

    for i in range(np.size(a)):
        cons.append({'type': 'ineq', 'fun': lambda a: max_a - a[i]})
        cons.append({'type': 'ineq', 'fun': lambda a: a[i] - min_a})

    return cons

def piecewise_objective(x_a_opt, *args):
    q1 = args[0]
    q2 = args[1]
    q3 = args[2]
    h = args[3]
    x = args[4]
    r = args[5] # The number of segmentations
    v = (x[1:] - x[:-1]) / h

    x_opt = x_a_opt[:-r]
    a_opt = x_a_opt[-r:]
    v_opt = [0] * (len(x_opt) - 1)
    r = len(a_opt)
    block = float(len(v_opt)) / r

    for i in range(len(v_opt)):
        v_opt[i] = (x_opt[i+1] - x_opt[i]) * a_opt[int(i//block)] / h

    x_delta = np.atleast_2d(x_opt - x)
    v_delta = np.atleast_2d(v_opt - v)
    a_delta = np.atleast_2d(a_opt - 1)

    return float(np.matmul(np.matmul(x_delta, q1), x_delta.T) + np.matmul(np.matmul(v_delta, q2), v_delta.T) + np.matmul(np.matmul(a_delta, q3), a_delta.T))


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

    def myobjective(x):
        print("Running")
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
    sol = minimize(myobjective, x0, method='SLSQP', constraints=[cons, len_cons, len_cons_2], options={'disp': True})