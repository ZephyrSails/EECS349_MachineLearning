#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import scipy.stats

from plot_helper import *


def main():
    """
    This function runs your code for problem 2.

    You can also use this to test your code for problem 1,
    but make sure that you do not leave anything in here that will interfere
    with problem 2. Especially make sure that gmm_est does not output anything
    extraneous, as problem 2 has a very specific expected output.
    """
    file_path = sys.argv[1]

    X1, X2 = read_gmm_file(file_path)
    # YOUR CODE FOR PROBLEM 2 GOES HERE

    # init_mu = np.array([0, 50])
    # init_sigmasq = np.array([3, 9])
    # init_wt = np.array([.5, .5])
    # mu, sigmasq, wt, L = gmm_est(np.concatenate((X1, X2)), init_mu, init_sigmasq, init_wt, 20, X1, X2)

    init_mu1 = np.array([10., 30.])
    init_sigmasq1 = np.array([8., 6.])
    init_wt1 = np.array([.6, .4])

    init_mu2 = np.array([-25., -5., 50.])
    init_sigmasq2 = np.array([3., 10., 20.])
    init_wt2 = np.array([.2, .5, .3])

    its = 20

    mu1, sigmasq1, wt1, L1 = gmm_est(X1, init_mu1, init_sigmasq1, init_wt1, its)
    mu2, sigmasq2, wt2, L2 = gmm_est(X2, init_mu2, init_sigmasq2, init_wt2, its)

    # (mu_results1, mu_results2), (sigma2_results1, sigma2_results2), (w_results1, w_results2), L = gmm_est(X1 + X2, array([1, 1]), array([0, 0]), array([0, 0]), 20)

    # mu_results1, sigma2_results1, w_results1 are all numpy arrays
    # with learned parameters from Class 1
    print 'Class 1'
    print 'mu =', mu1, '\nsigma^2 =', sigmasq1, '\nw =', wt1
    # mu_results2, sigma2_results2, w_results2 are all numpy arrays
    # with learned parameters from Class 2
    print '\nClass 2'
    print 'mu =', mu2, '\nsigma^2 =', sigmasq2, '\nw =', wt2

    plotClasses(X1, X2, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, save=True)
    # plot(np.concatenate((X1, X2)), mu, sigmasq, wt)
    # plotGMM(X1, X2, mu, sigmasq, wt)


def gmm_est(X, mu_init, sigmasq_init, wt_init, its, X1=None, X2=None):
    """
    Input Parameters:
      - X             : N 1-dimensional data points (a 1-by-N numpy array)
      - mu_init       : initial means of K Gaussian components (a 1-by-K numpy array)
      - sigmasq_init  : initial  variances of K Gaussian components (a 1-by-K numpy array)
      - wt_init       : initial weights of k Gaussian components (a 1-by-K numpy array that sums to 1)
      - its           : number of iterations for the EM algorithm

    Returns:
      - mu            : means of Gaussian components (a 1-by-K numpy array)
      - sigmasq       : variances of Gaussian components (a 1-by-K numpy array)
      - wt            : weights of Gaussian components (a 1-by-K numpy array, sums to 1)
      - L             : log likelihood
    """

    # YOUR CODE FOR PROBLEM 1 HERE
    mu = mu_init
    sigmasq = sigmasq_init
    wt = wt_init
    K = len(mu)
    L = sum(np.log(sum([wt[j] * scipy.stats.norm(mu[j], sigmasq[j] ** .5).pdf(X) for j in xrange(K)])))
    # print wt, mu, sigmasq, L
    # print ''
    for _ in xrange(its):
        wps = np.array([wt[j] * scipy.stats.norm(mu[j], sigmasq[j] ** .5).pdf(X) for j in xrange(K)])
        swps = sum(wps)

        r = wps / swps
        t = sum(r.T)
        wt      = t / len(X)
        mu      = np.dot(r, X) / t
        sigmasq = np.array([np.dot(r[j], (X - mu[j]) ** 2) for j in xrange(K)]) / t

        L = sum(np.log(sum([wt[j] * scipy.stats.norm(mu[j], sigmasq[j] ** .5).pdf(X) for j in xrange(K)])))

        # print wt, mu, sigmasq, L

    return mu, sigmasq, wt, L


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
