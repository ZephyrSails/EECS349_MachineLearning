# 943   Users
# 1682  Items
import numpy as np
import csv
import random


def read(fileName):
    numOfUsers = 943
    numOfItems = 1682
    matrix = np.zeros((numOfUsers+1, numOfItems+1))
    Users = []
    Iterms = []
    with open(fileName, 'r') as f:
        dat = csv.reader(f, delimiter='\t')
        for row in dat:
            matrix[int(row[0])][int(row[1])] = int(row[2])
    # print sum(matrix[1,:])
    return matrix


def readSample(fileName, testSet):
    numOfUsers = 943
    numOfItems = 1682
    matrix = np.zeros((numOfUsers+1, numOfItems+1))
    Users = []
    Iterms = []
    testQueries = []
    with open(fileName, 'r') as f:
        dat = csv.reader(f, delimiter='\t')
        for row in dat:
            if dat.line_num in testSet:
                testQueries.append((int(row[0]), int(row[1]), int(row[2])))
            else:
                matrix[int(row[0])][int(row[1])] = int(row[2])
    # print sum(matrix[1,:])
    return matrix, testQueries


#   This function is used to generate samples
#   since samples for each test should be the same
#   we need to store them to disk.
def generateSamples(sampleNum=50, entryNum=100, total=100000):
    matrix = np.zeros((sampleNum, entryNum))
    indexes = range(total)
    for i in xrange(sampleNum):
        random.shuffle(indexes)
        # currEntries = indexes
        for j in xrange(entryNum):
            matrix[i][j] = indexes[j]
    np.save('sampleSets', matrix)


if __name__ == '__main__':
    read('ml-100k/u.data')
