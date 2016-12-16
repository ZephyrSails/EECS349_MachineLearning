import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

def plotClasses(X1, X2, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, **kwargs):
    # (Y_test and X_test should be numpy arrays)

    X = np.array(range(-40, 100))
    Y1 = np.array(sum([wt1[j] * scipy.stats.norm(mu1[j], sigmasq1[j] ** .5).pdf(X) for j in xrange(len(mu1))]) * len(X1))
    Y2 = np.array(sum([wt2[j] * scipy.stats.norm(mu2[j], sigmasq2[j] ** .5).pdf(X) for j in xrange(len(mu2))]) * len(X2))
    Y = Y1 + Y2

    bins = 50 # the number 50 is just an example.

    plt.subplot(3,1,1)
    plt.hist(X1, bins, color='red', alpha=.5)
    plt.plot(X, Y1, color='red', alpha=.5)
    plt.title('Class 1')

    plt.subplot(3,1,2)
    plt.hist(X2, bins, color='blue', alpha=.5)
    plt.plot(X, Y2, color='blue', alpha=.5)
    plt.title('Class 2')

    plt.subplot(3,1,3)
    plt.hist([X1, X2], bins, color=['red', 'blue'], alpha=.5)
    plt.plot(X, Y, color='purple', alpha=.5, label='Combine')
    plt.plot(X, Y1, color='red', alpha=.5, label='Class 1')
    plt.plot(X, Y2, color='blue', alpha=.5, label='Class 2')
    plt.legend()
    plt.title('Combine')

    if kwargs.get('save', False):
        plt.savefig('likelihood_classes.png')
    else:
        plt.show()


def plotClassify(X1, X2, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, A1, A2, accuracy, **kwargs):
    # (Y_test and X_test should be numpy arrays)

    X = np.array(range(-40, 100))
    Y1 = np.array(sum([wt1[j] * scipy.stats.norm(mu1[j], sigmasq1[j] ** .5).pdf(X) for j in xrange(len(mu1))]) * len(X1))
    Y2 = np.array(sum([wt2[j] * scipy.stats.norm(mu2[j], sigmasq2[j] ** .5).pdf(X) for j in xrange(len(mu2))]) * len(X2))
    Y = Y1 + Y2

    bins = 50 # the number 50 is just an example.

    plt.subplot(4,1,1)
    plt.hist(X1, bins, color='red', alpha=.5)
    plt.plot(X, Y1, color='red', alpha=.5, label='Gaussian learned by Training Set')
    plt.legend()
    plt.title('Class 1 in training set')

    plt.subplot(4,1,2)
    plt.hist(X2, bins, color='blue', alpha=.5)
    plt.plot(X, Y2, color='blue', alpha=.5, label='Gaussian learned by Training Set')
    plt.legend()
    plt.title('Class 2 in training set')


    plt.subplot(4,1,3)
    plt.hist([X1, X2], bins, color=['red', 'blue'], alpha=.5, label=['Real Class 1', 'Real Class 2'])
    # plt.plot(X, Y, color='purple', alpha=.5, label='Combined Gaussian')
    plt.plot(X, Y1, color='red', alpha=.5, label='Class 1 Training Gaussian')
    plt.plot(X, Y2, color='blue', alpha=.5, label='Class 2 Training Gaussian')
    plt.legend()
    plt.title('Combined testing set')

    plt.subplot(4,1,4)
    plt.hist([A1, A2], bins, color=['red', 'blue'], alpha=.5, label=['Classified as 1', 'Classified as 2'])
    # plt.plot(X, Y, color='purple', alpha=.5, label='Combined Gaussian')
    plt.plot(X, Y1, color='red', alpha=.5, label='Class 1 Training Gaussian')
    plt.plot(X, Y2, color='blue', alpha=.5, label='Class 2 Training Gaussian')
    plt.legend()
    plt.title('Classification result, Accuracy: %f' % accuracy)


    if kwargs.get('save', False):
        plt.savefig('likelihood_classes.png')
    else:
        plt.show()

# python gmm_est.py 'gmm_train.csv'
def plotGMM(X1, X2, mu, sigmasq, wt):
    # (Y_test and X_test should be numpy arrays)
    class1 = X1
    class2 = X2
    # class1 = X_test[np.nonzero(Y_test == 1)[0]]
    # class2 = X_test[np.nonzero(Y_test == 2)[0]]
    bins = 50 # the number 50 is just an example.

    # plt.subplot(2,1,1)
    # plt.hist(class1, bins, color='blue')
    # plt.subplot(2,1,2)
    X = np.array(range(-40, 100))

    Y = sum([wt[j] * scipy.stats.norm(mu[j], sigmasq[j] ** .5).pdf(X) for j in xrange(len(mu))]) * (len(X1) + len(X2))

    plt.plot(X, Y)
    plt.hist([class2, class1], bins, color=['red', 'blue'], alpha=.5)
    # plt.hist(class1, bins, color='blue', alpha=.5)
    plt.show()


def plot(X1, mu, sigmasq, wt):
    X = np.array(range(-40, 100))

    Y = sum([wt[j] * scipy.stats.norm(mu[j], sigmasq[j] ** .5).pdf(X) for j in xrange(len(mu))]) * len(X1)

    plt.plot(X, Y)
    plt.hist(X1, 50, color='blue', alpha=.5)
    # plt.hist(class1, bins, color='blue', alpha=.5)
    plt.show()
