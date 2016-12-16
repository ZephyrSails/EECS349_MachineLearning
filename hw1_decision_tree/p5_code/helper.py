# Zhiping Xiu 2016-09-28 01:04:09 -0500
import numpy as np
import csv
import warnings
# The helper is written by Zhiping too
# But this might be reuseable in the future

# entries = load('IvyLeague.txt')
# entries = load('MajorityRule.txt')
# read file, return something like this: [{}, {}, ..., {}]
def load(fileName):
    entries = []
    with open(fileName, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # for key, value in row.items(): # I don't translate string to bool,
            #     if value == 'true':        # I think string more extendable,
            #         row[key] = True        # Please tell me why,
            #     else:                      # if you think I'm wrong.
            #         row[key] = False       # see also DecisionNode
            entries.append(row)
    return entries

# I explain how gain and entropy work by following example:
# Considering, we want to calculate the gain of this node decision:
###############################
#            S = [9+, 5-]     #
# values(wind) = weak, strong #
#       S_weak = [6+, 2-]     #
#     S_strong = [3+, 3-]     #
###############################
# >>> gain([9, 5], [[6, 2], [3, 3]])
# 0.048
# or you can use:
# >>> gain(entropy([9, 5]), [[6, 2], [3, 3]])
# 0.048
def gain(s, a):
    a = np.array(a)
    weights = 1.0 * np.sum(a, axis=1) / np.sum(a)
    if isinstance(s, list):
        s = np.array(s)
        s = entropy(s)
    return s - np.sum(weights * entropy(a))

# >>> entropy([9, 5])
# 0.940
# or you can use:
# >>> entropy([[6, 2], [3, 3]])
# array([ 0.811,  1.  ])
def entropy(array):
    arr = np.array(array)
    dim = arr.ndim
    warnings.filterwarnings("ignore")
    # following two line can raise following two worning:
    # divide by zero encountered in log2
    # and
    # invalid value encountered in divide
    # I choose to shut them up. But this maybe not a very smart way to do this
    prob_arr = np.transpose(1.0 * np.transpose(arr) / (np.sum(arr, axis=(dim-1))))
    log = np.log2(prob_arr)
    # But don't worry, it will be taken care in following two line:
    prob_arr[np.isinf(prob_arr) | np.isnan(prob_arr)] = 0
    log[np.isinf(log) | np.isnan(log)] = 0
    return (np.sum(prob_arr * log, axis=(dim-1)) * -1)
