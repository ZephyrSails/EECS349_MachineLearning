# Zhiping Xiu 2016-09-28 01:04:09 -0500
import csv
import numpy as np
from helper import *
import sys
import random
# USAGE:
# $ python decisiontree.py 'IvyLeague.txt' 16 1 0
# or
# $ python decisiontree.py 'MajorityRule.txt' '16' '2' '0'

class DecisionNode:
    def __init__(self, attr, sub, value=None, result=None):
        self.attr   = attr          # which column of data to evaluate here
        self.sub    = sub           # sub node dict {'true': None, 'false': None}
        self.value  = value         # value array
        self.result = result        # possible branch
    def show(self, layer=0, prefix=''):
        # prefix = '  '*layer + prefix + ''
        print '    '*layer + prefix + ' -> ' + self.attr + '?'
        layer += 1
        for key, sub_node in self.sub.iteritems():
            # print key, sub_node
            if isinstance(sub_node, str):
                print '    '*layer + key + ' -> ' + sub_node.upper() + '!'
            else:
                sub_node.show(layer, key)
        return

# split(entries, 'CLASS')
# return: [[{}, {}, ..., {}], [{}, {}, ..., {}]]
def split(entries, column, values=['true', 'false']):
    result = [[] for x in range(len(values))]
    for row in entries:
        for idx, val in enumerate(values):
            if row[column] == val:
                result[idx].append(row)
    return result

# USAGE:
# attributes = ['GoodGrades', 'GoodLetters', 'GoodSAT', 'IsRich', 'HasScholarship', 'ParentAlum', 'SchoolActivities']
# root = build_tree(entries, attributes)
# Build the decision tree!
def build_tree(entries=[{}], attributes=[], node=False, values=['true', 'false']):
    s = [len(x) for x in split(entries, 'CLASS')]
    ss = sum(s)
    for idx, val in enumerate(s): # If entries are fully divided, job done
        if val == ss:
            return values[idx]
    if len(attributes) == 0: # If there are no attributes left, job can't continue
        return values[s.index(max(s))] # return the best guess we got
    s = entropy(s)
    most_gain = 0
    best_column = attributes[0]
    for column in attributes:
        g = gain(s, [[len(z) for z in y] # count length, axis=2: [[1, 1], [1, 1]]
                        for y in [split(x, 'CLASS') # split each column by 'CLASS': [[[{}],[{}]],[[{}],[{}]]]
                        for x in split(entries, column)]]) # split by column: [[{},{}],[{},{}]]
        if g > most_gain:
            most_gain = g
            best_column = column
    # build most gain node, recursively find it's sub branch
    splitted_entries = dict(zip(values, split(entries, best_column)))
    node = DecisionNode(best_column, dict(zip(values, len(values) * [None])))
    attributes.remove(best_column)
    for key in node.sub:
        node.sub[key] = build_tree(splitted_entries[key], attributes)
    attributes.append(best_column)
    return node

# Function we used to make prediction. Based on decision tree we built
def hypo(tree_node, entry, prior=-1):
    if isinstance(tree_node, str): # Leaf found
        return tree_node
    return hypo(tree_node.sub[entry[tree_node.attr]], entry)

def split_training_testing_set(entries, trainingSetSize):
    random.shuffle(entries)
    return [entries[:trainingSetSize], entries[trainingSetSize:]]

# How often does the yes condition actually occur in our sample?
def get_prevalence(entries):
    return 1.0 * len([i for i in entries if i['CLASS'] == 'true']) / len(entries)

# Overall, how often is the prior classifier correct?
def get_accuracy_by_prior(testing_set, prior, verbose='0'):
    if verbose == '1':
        print '------- The classification returned by prior of training set begin -------'
    correct_count = 0
    for entry in testing_set:
        if verbose=='1':
            if prior > .5:
                print '\ttrue by prior, for ' + str(entry)
            else:
                print '\tfalse by prior, for ' + str(entry)
            # if random.random() < prior:
            #     print '\ttrue by prior, for ' + str(entry)
            # else:
            #     print '\tfalse by prior, for ' + str(entry)
        # if ((entry['CLASS'] == 'true') and (random.random() < prior)) or ((entry['CLASS'] == 'false') and (random.random() > prior)):
        if ((entry['CLASS'] == 'true') and (prior > .5)) or ((entry['CLASS'] == 'false') and (prior <= .5)):
            correct_count += 1
    return 1.0 * correct_count / len(testing_set)

# Overall, how often is the decision tree classifier correct?
def get_accuracy(testing_set, tree_root, verbose='0'):
    if verbose == '1':
        print '------- The classification returned by decision tree for begin -------'
    correct_count = 0
    for entry in testing_set:
        hypo_result = hypo(tree_root, entry)
        if verbose=='1':
            print '\t' + hypo_result + ' by tree, for ' + str(entry)
        if entry['CLASS'] == hypo_result:
            correct_count += 1
    return 1.0 * correct_count / len(testing_set)

# inputFileName     - an integer specifying the number of examples from the
#                     input file that will be used to train the system
# trainingSetSize   - the fully specified path to the input file.
# numberOfTrials    - an integer specifying how many times a decision tree will
#                     be built from a randomly selected subset of the training
#                     examples.
# verbose           - a string that must be either '1' or '0'
#                     If verbose is '1' the output will include the training
#                     and test sets. Else the output will only contain a
#                     description of the tree structure and the results for the
#                     trials.
# Use this phrase to call:
# $ python decisiontree.py 'IvyLeague.txt' 16 1 0
# $ python decisiontree.py 'MajorityRule.txt' 16 1 0
def decisiontree(inputFileName, trainingSetSize='16', numberOfTrails='1', verbose='0'):
    data_set = load(inputFileName)
    print 'The input file name:   ' + inputFileName
    print 'The training set size: ' + trainingSetSize
    print 'The testing set size:  ' + str(len(data_set)-int(trainingSetSize))
    if (len(data_set)-int(trainingSetSize)) <= 0:
        print 'ERROR: respectfully, I disagree with the size of your testing set.'
        return
    print 'The number of trials: ' + numberOfTrails
    # if int(trainingSetSize) >= len(data_set))
    sum_accuracy = 0
    sum_accuracy_by_prior = 0
    for i in range(1, int(numberOfTrails)+1):
        print '------- Trail ' + str(i) + ' -------'
        [training_set, testing_set] = split_training_testing_set(data_set, int(trainingSetSize))
        prior = get_prevalence(training_set)
        print 'Testing set prior: ' + str(prior)
        attributes = training_set[0].keys()
        attributes.remove('CLASS')
        root = build_tree(training_set, attributes)
        print 'The structure of the decision tree:'
        root.show()
        if verbose == '1':
            print '------- Verbose begin for Trail ' + str(i) + ' -------'
            print 'training_set:'
            print training_set
            print 'testing_set:'
            print testing_set
        accuracy = get_accuracy(testing_set, root, verbose)
        accuracy_by_prior = get_accuracy_by_prior(testing_set, prior, verbose)
        sum_accuracy += accuracy
        sum_accuracy_by_prior += accuracy_by_prior
        if verbose == '1':
            print '------- Verbose ended for Trail ' + str(i) + ' -------'
        print 'Accuracy by decision tree: ' + str(accuracy)
        print 'Accuracy by testing set prior: ' + str(accuracy_by_prior)

    print '-------' + numberOfTrails + ' times of Trail ended -------'
    print 'The mean classification performance by decision tree: ' + str(1.0 * sum_accuracy / int(numberOfTrails))
    print 'The mean classification performance by testing set prior: ' + str(1.0 * sum_accuracy_by_prior / int(numberOfTrails))
    # collection = helper.load(inputFileName)

if __name__ == "__main__":
    decisiontree(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    # I'm not sure about this:
    # But this is ugly, and unextenable,
    # Consider I want to use default value '0' for verbose, I can't do that
    #                         --> IndexError: list index out of range
    # Unless I add some judgement to check is certain parameters are here or not
    # is there better way?
