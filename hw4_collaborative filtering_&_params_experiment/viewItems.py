from loader import *
import matplotlib.pyplot as plt
import numpy as np
import time
# 1.B:
#   For each movie:
#       How many reviews it has?
#   Which movie has most reviews? - How many?
#   What has fewest reviews?
#   Order the movies by number of reviews
#   Plot:
#       x: movies number ordered by reviews
#       y: number of reviews
def visualizeMovies(fileName):
    matrix = read(fileName)
    reviewCountMatrix = countReviews(matrix)
    printInfo(reviewCountMatrix)
    plot(reviewCountMatrix)
    plotZipf(reviewCountMatrix)


def printInfo(reviewCountMatrix):
    print len(reviewCountMatrix)
    mostReviewedMovie = max(list(enumerate(reviewCountMatrix)), key=lambda x: x[1])
    leastReviewedMovie = min(list(enumerate(reviewCountMatrix[1:])), key=lambda x: x[1])
    print 'Movie ID who has most reviews %d, it has %d reviews' % (mostReviewedMovie[0], mostReviewedMovie[1])
    print 'Movie ID who has least reviews %d, it has %d reviews' % (leastReviewedMovie[0]+1, leastReviewedMovie[1])


def plot(reviewCountMatrix):
    itemCount = np.shape(reviewCountMatrix)[0]
    plt.plot(range(1, itemCount), sorted(reviewCountMatrix[1:]))
    plt.xlabel('movie')
    plt.ylabel('number of reviews')
    plt.title('movies, sorted by number of reviews')
    plt.show()

def plotZipf(reviewCountMatrix):
    itemCount = np.shape(reviewCountMatrix)[0]
    plt.scatter(np.log(np.array(range(1, itemCount))), np.log(sorted(reviewCountMatrix[1:], reverse=True)))
    plt.xlabel('log movies rank')
    plt.ylabel('log number of reviews')
    plt.title('movies, sorted by number of reviews log-log Chart')
    plt.show()


def countReviews(matrix):
    itemCount = np.shape(matrix)[1]
    userCount = np.shape(matrix)[0]

    return np.array([sum([True for i in matrix[:, j] if i > 0])
                    for j in xrange(itemCount)])


if __name__ == '__main__':
    startAt = time.time()
    visualizeMovies('ml-100k/u.data')
    print 'Used time: %f' % (time.time() - startAt)
