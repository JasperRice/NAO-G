import matplotlib.pyplot as plt
import numpy as np

Xs = [20 + 10 * j for j in range(11)]

files = [open("size_test.csv", 'r')]
file_01 = open("dataset/size_test_01.csv", 'r')
file_02 = open("dataset/size_test_02.csv", 'r')
file_03 = open("dataset/size_test_03.csv", 'r')
file_04 = open("dataset/size_test_04.csv", 'r')
file_05 = open("dataset/size_test_05.csv", 'r')
# files = [file_01, file_02, file_03, file_04, file_05]

temp = [[] for _ in range(len(Xs))]
y = []
e = []

for file in files:
    lines = file.readlines()
    for i, line in enumerate(lines):
        wordList = line.split(', ')
        temp[i] += list(map(float, wordList))
    file.close()

for t in temp:
    t_arr = np.array(t)
    y.append(np.mean(t_arr))
    e.append(np.std(t_arr))

x = np.array(Xs)
y = np.array(y)
e = np.array(e)

plt.errorbar(x, y, e, linestyle='--', marker='.', fmt='-o')
plt.xlabel("Number of training data")
plt.ylabel("Test error with standard deviation")
plt.show()
