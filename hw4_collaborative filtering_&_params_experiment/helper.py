import scipy.stats
import scipy.spatial.distance
import numpy as np
from scipy.stats import mode

def manhattanDistance(u, v):
    return scipy.spatial.distance.cityblock(u, v)


def pearsonrDistance(u, v):
    return scipy.stats.pearsonr(u, v)[0]


def kNNRating(rowsId, columnId, matrix, distance, k, iFlag):
    u = matrix[rowsId]
    if distance:
        distanceMatrix = sorted([(manhattanDistance(u, matrix[index_v]), index_v) for index_v in xrange(1, len(matrix))])
    else:
        distanceMatrix = sorted([(pearsonrDistance(u, matrix[index_v]), index_v) for index_v in xrange(1, len(matrix))], reverse=True)
    # kNNMatrix = np.array([])

    # print len(distanceMatrix)
    sumRate = 0
    kAns = []
    i, j = 0, 0
    # There probably no k valid entry at all! if so
    # we print as much valid answer as possible.
    # print distanceMatrix[:10]
    while i < k and j < len(distanceMatrix):
        # print j
        jId = distanceMatrix[j][1]
        if jId != rowsId and (iFlag or (matrix[jId][columnId] != 0)):
            # print jId, columnId
            sumRate += matrix[jId][columnId]
            kAns.append(matrix[jId][columnId])
            # print matrix[jId][movieid], distanceMatrix[j][0]
            i += 1
        j += 1
    # print i
    return mode(kAns)[0][0]

    
    #
    # return sumRate / i if i else 0
