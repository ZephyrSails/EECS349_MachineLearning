import numpy as np
from helper import *
from loader import *
import sys
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

#   pValue('evaResult/4_CD.npy', 0, 1)
def pValue(fileName, col1, col2):
    results = np.load(fileName)
    t, p = ttest_ind(results[0], results[1])
    # print p
    # print str(p)
    print 'p-value: %s' % str(p)


def plot4_CD():
    results = np.load('evaResult/4_CD.npy')
    plt.boxplot(results.transpose())
    settingSet = [  (0, 8, 0, 1),   # Pearson,      non-0 only,     item-based - shown better on test
                    (1, 8, 0, 1),   # Manhattan,    non-0 only,     item-based
                    (0, 8, 1, 1),   # Pearson,      all KNN,        item-based
                    (1, 8, 1, 1),   # Manhattan,    all KNN,        item-based
                    (0, 8, 0, 0),   # Pearson,      non-0 only,     user-based
                    (1, 8, 0, 0),   # Manhattan,    non-0 only,     user-based - shown better on test
                    (0, 8, 1, 0),   # Pearson,      all KNN,        user-based
                    (1, 8, 1, 0)]   # Manhattan,    all KNN,        user-based
    plt.xticks(range(1, 9), settingSet)
    plt.show()


def plot4_EF():
    results = np.load('evaResult/4_EF.npy')
    plt.boxplot(results.transpose())
    settingSet = [  (0, 1, 0, 1),   # item-based,   Pearson
                    (0, 2, 0, 1),   # item-based,   Pearson
                    (0, 4, 0, 1),   # item-based,   Pearson
                    (0, 8, 0, 1),   # item-based,   Pearson
                    (0, 16, 0, 1),  # item-based,   Pearson
                    (0, 32, 0, 1),  # item-based,   Pearson
                    (1, 1, 0, 0),   # user-based,   Manhattan
                    (1, 2, 0, 0),   # user-based,   Manhattan
                    (1, 4, 0, 0),   # user-based,   Manhattan
                    (1, 8, 0, 0),   # user-based,   Manhattan
                    (1, 16, 0, 0),  # user-based,   Manhattan
                    (1, 32, 0, 0)]  # user-based,   Manhattan
    plt.xticks(range(1, 13), settingSet)
    plt.show()

#   Set the parameters you want to compare here
def batchEva():
    # 4-C & D: choice for <distance>  and <i>
    # settingSet = [  (0, 8, 0, 1),   # Pearson,      non-0 only,     item-based
    #                 (1, 8, 0, 1),   # Manhattan,    non-0 only,     item-based
    #                 (0, 8, 1, 1),   # Pearson,      all KNN,        item-based
    #                 (1, 8, 1, 1),   # Manhattan,    all KNN,        item-based
    #                 (0, 8, 0, 0),   # Pearson,      non-0 only,     user-based
    #                 (1, 8, 0, 0),   # Manhattan,    non-0 only,     user-based
    #                 (0, 8, 1, 0),   # Pearson,      all KNN,        user-based
    #                 (1, 8, 1, 0)]   # Manhattan,    all KNN,        user-based
    # mses = np.array([eva('ml-100k/u.data', setting[0], setting[1], setting[2], setting[3])
    #                 for setting in settingSet])
    # np.save('evaResult/4_CD', mses)

    # 4-E & F: choice for <i>, item-based vs user-based
    settingSet = [  (0, 1, 0, 1),   # item-based,   Pearson
                    (0, 2, 0, 1),   # item-based,   Pearson
                    (0, 4, 0, 1),   # item-based,   Pearson
                    (0, 8, 0, 1),   # item-based,   Pearson
                    (0, 16, 0, 1),  # item-based,   Pearson
                    (0, 32, 0, 1),  # item-based,   Pearson
                    (1, 1, 0, 0),   # user-based,   Manhattan
                    (1, 2, 0, 0),   # user-based,   Manhattan
                    (1, 4, 0, 0),   # user-based,   Manhattan
                    (1, 8, 0, 0),   # user-based,   Manhattan
                    (1, 16, 0, 0),  # user-based,   Manhattan
                    (1, 32, 0, 0)]  # user-based,   Manhattan
    mses = np.array([eva('ml-100k/u.data', setting[0], setting[1], setting[2], setting[3])
                    for setting in settingSet])
    np.save('evaResult/4_EF', mses)

    # settingSet = [(0, 8, 0, 0), (0, 8, 0, 0)]
    # mses = np.array([eva('ml-100k/u.data', setting[0], setting[1], setting[2], setting[2]) for setting in settingSet])
    # np.save('evaResult/4_D', mses)
    # # 4-F

def eva(datafile, distance, k, iFlag, isItemBasedCf):
    sampleSets = np.load('sampleSets.npy')
    MSEs = np.array([])
    for sample in sampleSets:
        MSE = sampleTest(datafile, isItemBasedCf, set(sample), distance, k, iFlag)
        MSEs = np.append(MSEs, MSE)
        print MSE
    return MSEs


def sampleTest(datafile, isItemBasedCf, sampleSet, distance, k, iFlag):
    matrix, queries = readSample(datafile, sampleSet)

    if isItemBasedCf:
        matrix = matrix.transpose()
    sumSquaredError = 0

    for userid, movieid, trueRating in queries:
        if isItemBasedCf:
            # print userid, movieid
            predictedRating = kNNRating(movieid, userid, matrix, distance, k, iFlag)
        else:
            # print userid, movieid
            predictedRating = kNNRating(userid, movieid, matrix, distance, k, iFlag)
        sumSquaredError += pow(predictedRating - trueRating, 2)
    return sumSquaredError / len(queries)


# <datafile> - a fully specified path to a file formatted like the MovieLens100K data file u.data
# <distance> - a Boolean. If set to 0, use Pearson's correlation as the distance measure. If 1, use Manhattan distance.
# <k> - The number of nearest neighbors to consider
# <iFlag> - A Boolean value. If set to 0 for user-based collaborative filtering,
# only users that have actual (ie non-0) ratings for the movie are considered in your top K.
# For item-based, use only movies that have actual ratings by the user in your top K.
# If set to 1, simply use the top K regardless of whether the top K contain actual or filled-in ratings.
# <isItemBasedCf> - a Boolean, if set to 1, use item-based cf method, if set to 0, use user-based
# Usage:
# python eva.py
if __name__ == '__main__':
    # datafile = sys.argv[1]
    # distance = int(sys.argv[2])
    # k = int(sys.argv[3])
    # i = int(sys.argv[4])
    # isItemBasedCf = int(sys.argv[5])
    # eva(datafile, distance, k, i, isItemBasedCf)
    batchEva()
