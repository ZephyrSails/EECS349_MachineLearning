from spamfilter import *
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def BayesVSPrior():
    print 'Bayes\'s turn'
    bayesErrorRates = cv('sourceDataSet', 'resultDirBayes', 10, 0)
    print 'Prior\'s turn'
    PriorErrorRates = cv('sourceDataSet', 'resultDirPrior', 10, 1)
    print 'Mean error rate for Bayes Classifier: %f' % (float(sum(bayesErrorRates)) / len(bayesErrorRates))
    print 'Mean error rate for Prior Classifier: %f' % (float(sum(PriorErrorRates)) / len(PriorErrorRates))
    pValue(bayesErrorRates, PriorErrorRates)

    plotErrorRates(bayesErrorRates, PriorErrorRates)

def pValue(col1, col2):
    # results = np.load(fileName)
    t, p = ttest_ind(col1, col2)
    # print p
    # print str(p)
    print 'p-value: %s' % str(p)


def plotErrorRates(Y1, Y2):
    plt.plot(range(len(Y1)), Y1, color='red', label='bayes error rate')
    plt.plot(range(len(Y2)), Y2, color='blue', label='prior error rate')
    plt.xlabel('cross validation genration')
    plt.ylabel('error rate')
    plt.legend()
    plt.title('Bayes VS Prior')
    plt.show()


if __name__ == '__main__':
    BayesVSPrior()
