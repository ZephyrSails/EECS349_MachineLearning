import sys
import numpy as np
from loader import *
import matplotlib.pyplot as plt
import time
# 1.A:
#   For each pair of users:
#       How many common reviewed movie
#   Mean?   Median?
#   Plot:
#       x: common movie count
#       y: pair count who has reviewed x common movie
def visualizeUsers(fileName):
    # This is wrong
    # matrix = read(fileName)
    # itemPairs = pairCount(matrix)
    # np.save('itemPairs', itemPairs)
    # itemPairs = np.load('itemPairs.npy')
    # printInfo(itemPairs)
    # plot(itemPairs)


    # Un-comment following three line, to count pairs again
    # matrix = read(fileName)
    # pairCommonReviewVector = pairCommonReview(matrix)
    # np.save('UserPairs', pairCommonReviewVector)
    pairCommonReviewVector = np.load('UserPairs.npy')
    print 'Mean: %f' % (float(sum(pairCommonReviewVector)) / len(pairCommonReviewVector))
    print 'Median: %d' % pairCommonReviewVector[len(pairCommonReviewVector)/2]
    preProcessForBarChart(pairCommonReviewVector)
    # print pairCommonReviewVector[0], pairCommonReviewVector[-1]
    barVector = np.load('UserBars.npy')
    # print barVector[0], barVector[-1], len(barVector)
    plot(barVector)


def preProcessForBarChart(vector):
    barData = [0 for i in xrange(max(vector)+1)]
    for i in vector:
        barData[i] += 1
    np.save('UserBars', np.array(barData))


def plot(vector):
    plt.bar(range(len(vector)), vector)
    plt.xlabel('Common Reviewed Movies Count')
    plt.ylabel('Paris of common reviewers Count')
    plt.title('Paris of common reviewers of Movies')
    plt.show()


def pairCommonReview(matrix):
    userCount = np.shape(matrix)[0]
    itemCount = np.shape(matrix)[1]

    itemPairs = np.zeros((itemCount))
    ans = []
    for i in xrange(1, userCount):
        for j in xrange(i, userCount):
            count = 0
            for k in xrange(1, itemCount):
                if matrix[i][k] > 0 and matrix[j][k] > 0:
                    count += 1
            ans.append(count)
    return np.array(sorted(ans))


def printInfo(itemPairs):
    sortedItemParis = sorted(enumerate(itemPairs[1:]), key=lambda x: x[1])
    length = len(sortedItemParis)
    print length
    print itemPairs[0]
    print 'Mean common reviews count: %f' % (sum(itemPairs) / length)
    print 'Median common reviews count: %f' % ((sortedItemParis[length/2][1] + sortedItemParis[length/2+1][1]) / 2)


def pairCount(matrix):
    userCount = np.shape(matrix)[0]
    itemCount = np.shape(matrix)[1]

    itemPairs = np.zeros((itemCount))
    for i in xrange(1, userCount):
        for j in xrange(i, userCount):
            for k in xrange(1, itemCount):
                if matrix[i][k] > 0 and matrix[j][k] > 0:
                    itemPairs[k] += 1
    return itemPairs


# def plot(itemPairs):
#     count = np.shape(itemPairs)[0]-1
#     # plt.hist(np.linspace(1, count, count), sorted(itemPairs[1:]))
#     plt.bar(range(count), sorted(itemPairs[1:]))
#     # plt.plot(np.linspace(1, count, count), sorted(itemPairs[1:]))
#     plt.xlabel('Movies')
#     plt.ylabel('Paris of common reviewers')
#     plt.title('Paris of common reviewers of Movies')
#     plt.show()


if __name__ == '__main__':
    startAt = time.time()
    # datafile = sys.argv[1] # 'ml-100k/u.data'
    visualizeUsers('ml-100k/u.data')
    print 'Used time: %f' % (time.time() - startAt)
