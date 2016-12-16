import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
import itertools
import random
import sys
import classifier_1
import classifier_2
import os
from scipy.stats import ttest_ind


def cv(k, method, directory='.'):
    """
    :type directory: String
    :type k: Int
    :rtype: None
    """
    images, labels = load_mnist(digits=range(0, 10), path=directory)

    images = preprocess(images)
    groupedPairs = cvDivide(images, labels, k)
    erros_1 = []
    erros_2 = []

    for i in xrange(k):
        # trainingImages = np.concatenate(groupedImges[:i], groupedImges[:i]
        testingPairs = zip(*groupedPairs[i])
        trainingPairs = zip(*list(itertools.chain(*groupedPairs[:i])) + list(itertools.chain(*groupedPairs[i+1:])))


        testingImages = np.array(testingPairs[0])
        testingLabels = np.array(testingPairs[1])
        trainingImages = np.array(trainingPairs[0])
        trainingLabels = np.array(trainingPairs[1])
        # testingImages = testingPairs[0]
        # testingLabels = testingPairs[1]
        # trainingImages = trainingPairs[0]
        # trainingLabels = trainingPairs[1]

        print np.shape(testingImages), np.shape(testingLabels), np.shape(trainingImages), np.shape(trainingLabels)
        # if method == 0:
        erros_1.append(classifier_1.test(trainingImages, trainingLabels, testingImages, testingLabels)[0])
        # elif method == 1:
        erros_2.append(classifier_2.test(trainingImages, trainingLabels, testingImages, testingLabels)[0])

    print erros_1, erros_2


def cvDivide(datas, labels, k):
    """
    :type images:           List of images
    :type labels:           List of labels
    :type k:                K fold cross validation
    :rtype groupedPairs:    K Lists of evenly divided grouped data-lable pairs
    """
    pairs = zip(datas, labels)
    random.shuffle(pairs)

    return list(chunks(pairs, k))


# def eva(classfied, actualLabels):
#     acc =


# evenly divid a list to n part
def chunks(lst, n):
    l = [(float(len(lst))/n) * i for i in xrange(n+1)]
    for f, t in zip(l, l[1:]):
        yield lst[int(f):int(t)]


def preprocess(images):
    #this function is suggested to help build your classifier.
    #You might want to do something with the images before
    #handing them to the classifier. Right now it does nothing.
    return [i.flatten() for i in images]


def pltbox(result):
    # [0.018833333333333334, 0.018166666666666668, 0.018166666666666668, 0.017333333333333333, 0.017666666666666667, 0.017333333333333333, 0.016, 0.016166666666666666, 0.015666666666666666, 0.018833333333333334]
    # [0.027666666666666666, 0.029666666666666668, 0.025833333333333333, 0.03, 0.027, 0.029, 0.024666666666666667, 0.0255, 0.023, 0.027666666666666666]
    plt.boxplot(result)
    plt.ylim([0, 0.04])

    t, p = ttest_ind(result[0], result[1])
    plt.title('p-value: %s' % str(p))

    plt.show()

if __name__ == '__main__':
    # cv(10, int(sys.argv[1]), '.')
    # result = [[0.3, 0.31, 0.35, 0.26, 0.28, 0.15, 0.3, 0.31, 0.33, 0.28], [0.29, 0.2, 0.21, 0.24, 0.16, 0.07, 0.18, 0.2, 0.19, 0.21]]
    # result = [[0.018833333333333334, 0.018166666666666668, 0.018166666666666668, 0.017333333333333333, 0.017666666666666667, 0.017333333333333333, 0.016, 0.016166666666666666, 0.015666666666666666, 0.018833333333333334], [0.027666666666666666, 0.029666666666666668, 0.025833333333333333, 0.03, 0.027, 0.029, 0.024666666666666667, 0.0255, 0.023, 0.027666666666666666]]
    pltbox(result)
