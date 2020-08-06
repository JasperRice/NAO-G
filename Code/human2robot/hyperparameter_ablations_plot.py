import matplotlib.pyplot as plt
import numpy as np

ablation = 'denormalized_activation_functions'
file_train = open('ablations_'+ablation+'_results_train.csv', 'r')
file_val = open('ablations_'+ablation+'_results_val.csv', 'r')
file_test = open('ablations_'+ablation+'_results_test.csv', 'r')
x = [0, 0.0765078199812]

train = [[] for _ in range(len(x))];    y_train = [];   e_train = [];
val = [[] for _ in range(len(x))];      y_val = [];     e_val = []
test = [[] for _ in range(len(x))];     y_test = [];    e_test = []

lines_train = file_train.readlines()
for i, line in enumerate(lines_train):
    wordList = line.split(', ')
    train[i] += list(map(float, wordList))
file_train.close()

lines_val = file_val.readlines()
for i, line in enumerate(lines_val):
    wordList = line.split(', ')
    val[i] += list(map(float, wordList))
file_val.close()

lines_test = file_test.readlines()
for i, line in enumerate(lines_test):
    wordList = line.split(', ')
    test[i] += list(map(float, wordList))
file_test.close()

for t_train, t_val, t_test in zip(train, val, test):
    t_train_arr = np.array(t_train); y_train.append(np.mean(t_train_arr)); e_train.append(np.std(t_train_arr))
    t_val_arr = np.array(t_val); y_val.append(np.mean(t_val_arr)); e_val.append(np.std(t_val_arr))
    t_test_arr = np.array(t_test); y_test.append(np.mean(t_test_arr)); e_test.append(np.std(t_test_arr))

fig, ax = plt.subplots(1,1)
start = 0
end = len(x)
ax.errorbar(x[start:end], y_train[start:end], e_train[start:end], linestyle='--', marker='.', fmt='-o', label='Training Error')
ax.errorbar(x[start:end], y_val[start:end], e_val[start:end], linestyle='--', marker='.', fmt='-o', label='Validation Error')
ax.errorbar(x[start:end], y_test[start:end], e_test[start:end], linestyle='--', marker='.', fmt='-o', label='Test Error')
# plt.xscale('log')
# labels = ['With PCA', 'Without PCA']
# plt.xticks(x[start:end], labels[start:end])
plt.xlabel("Dropout Rate")
plt.ylabel("Error with Standard Deviation / rad")
plt.legend(loc='center right', bbox_to_anchor=(1, 0.3))
plt.show()