import matplotlib.pyplot as plt
import numpy as np

file = open("size_test.csv", 'r')
lines = file.readlines()
y = []
e = []
for line in lines:
    wordList = line.split(', ')
    testResults = np.array(map(float, wordList))
    y.append(np.mean(testResults))
    e.append(np.std(testResults))

x = np.array([25, 50, 75, 100, 125])
y = np.array(y)
e = np.array(e)

plt.errorbar(x, y, e, linestyle='None', marker='.', fmt='-o')
plt.xlabel("Number of training data")
plt.ylabel("Test error with standard deviation")
plt.show()
