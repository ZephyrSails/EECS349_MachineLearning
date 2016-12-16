#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import scipy.stats
from gmm_est import gmm_est
from plot_helper import *

def main():
    """
    This function runs your code for problem 3.

    You can use this code for problem 4, but make sure you do not
    interfere with what you need to do for problem 3.
    """
    file_path = sys.argv[1]

    # YOUR CODE FOR PROBLEM 3 GOES HERE
    T1, T2 = read_gmm_file(file_path)
    # print list(T1)
    X1, X2 = read_gmm_file('gmm_train.csv')

    # plotClasses(X1, X2)

    p1 = float(len(X1)) / (len(X1)+len(X2))

    init_mu1 = np.array([10., 30.])
    init_sigmasq1 = np.array([8., 6.])
    init_wt1 = np.array([.6, .4])

    init_mu2 = np.array([-25., -5., 50.])
    init_sigmasq2 = np.array([3., 10., 20.])
    init_wt2 = np.array([.2, .5, .3])


    its = 20

    mu1, sigmasq1, wt1, L1 = gmm_est(X1, init_mu1, init_sigmasq1, init_wt1, its)
    mu2, sigmasq2, wt2, L2 = gmm_est(X2, init_mu2, init_sigmasq2, init_wt2, its)

    # Random selecting testing array
    X = np.concatenate((T1, T2))
    # X = np.random.rand(14) * 140 - 40

    predicate = gmm_classify(X, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1)


    # class1_data = np.array([X[i] for i in xrange(len(X)) if predicate[i] == 1])
    # class2_data = np.array([X[i] for i in xrange(len(X)) if predicate[i] == 2])

    class1_data = X[np.nonzero(predicate == 1)[0]]
    class2_data = X[np.nonzero(predicate == 2)[0]]

    # print predicate
    # print np.array(X)

    # print class2_data

    # print np.array(list(X))

    # class1_data is a numpy array containing
    # all of the data points that your gmm classifier
    # predicted will be in class 1.
    print 'Class 1'
    print class1_data

    # class2_data is a numpy array containing
    # all of the data points that your gmm classifier
    # predicted will be in class 2.
    print '\nClass 2'
    print class2_data


def testOnTestingSet(mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, X1, X2, p1):
    T1, T2 = read_gmm_file('gmm_test.csv')

    # predicate = gmm_classify(np.concatenate((T1, T2)), mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1)
    predicate1 = gmm_classify(T1, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1)
    predicate2 = gmm_classify(T2, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1)


    true_class1_data = [T1[i] for i in xrange(len(T1)) if predicate1[i] == 1]
    false_class2_data = [T1[i] for i in xrange(len(T1)) if predicate1[i] == 2]
    true_class2_data = [T2[i] for i in xrange(len(T2)) if predicate2[i] == 2]
    false_class1_data = [T2[i] for i in xrange(len(T2)) if predicate2[i] == 1]
    accuracy = float(len(true_class1_data) + len(true_class2_data)) / (len(T1) + len(T2))

    plotClassify(T1, T2, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, np.concatenate((true_class1_data, false_class1_data)), np.concatenate((true_class2_data, false_class2_data)), accuracy)


def gmm_classify(X, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1):
    """
    Input Parameters:
        - X           : N 1-dimensional data points (a 1-by-N numpy array)
        - mu1         : means of Gaussian components of the 1st class (a 1-by-K1 numpy array)
        - sigmasq1    : variances of Gaussian components of the 1st class (a 1-by-K1 numpy array)
        - wt1         : weights of Gaussian components of the 1st class (a 1-by-K1 numpy array, sums to 1)
        - mu2         : means of Gaussian components of the 2nd class (a 1-by-K2 numpy array)
        - sigmasq2    : variances of Gaussian components of the 2nd class (a 1-by-K2 numpy array)
        - wt2         : weights of Gaussian components of the 2nd class (a 1-by-K2 numpy array, sums to 1)
        - p1          : the prior probability of class 1.

    Returns:
        - class_pred  : a numpy array containing results from the gmm classifier
                        (the results array should be in the same order as the input data points)
    """

    Y1 = sum([wt1[j] * scipy.stats.norm(mu1[j], sigmasq1[j] ** .5).pdf(X) for j in xrange(len(mu1))]) * p1
    Y2 = sum([wt2[j] * scipy.stats.norm(mu2[j], sigmasq2[j] ** .5).pdf(X) for j in xrange(len(mu2))]) * (1 - p1)

    # YOUR CODE FOR PROBLEM 3 HERE

    return np.array([1 if Y1[i] > Y2[i] else 2 for i in xrange(len(X))])


def read_gmm_file(path_to_file):
    """
    Reads either gmm_test.csv or gmm_train.csv
    :param path_to_file: path to .csv file
    :return: two numpy arrays for data with label 1 (X1) and data with label 2 (X2)
    """
    X1 = []
    X2 = []

    data = open(path_to_file).readlines()[1:] # we don't need the first line
    for d in data:
        d = d.split(',')

        # We know the data is either class 1 or class 2
        if int(d[1]) == 1:
            X1.append(float(d[0]))
        else:
            X2.append(float(d[0]))

    X1 = np.array(X1)
    X2 = np.array(X2)

    return X1, X2


if __name__ == '__main__':
    main()
