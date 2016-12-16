# import packages (including scikit-learn packages)
# a
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier # Use this function for adaboosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn import metrics
from mnist import load_mnist
from helper import *


def boosting_A(training_set, training_labels, testing_set, testing_labels):
	'''
	Input Parameters:
		- training_set: a 2-D numpy array that contains training examples (size: the number of training examples X the number of attributes)
		  (NOTE: If a training example is 10x10 images, the number of attributes will be 100. You need to reshape your training example)
		- training_labels: a 1-D numpy array that labels of training examples (size: the number of training examples)
		- testing_set: a 2-D numpy array that contains testing examples  (size: the number of testing examples X the number of attributes)
		- testing_labels: a 1-D numpy array that labels of testing examples (size: the number of testing examples)

	Returns:
		- predicted_labels: a 1-D numpy array that contains the labels predicted by the classifier. Labels in this array should be sorted in the same order as testing_labels
		- confusion_matrix: a 2-D numpy array of confusion matrix (size: the number of classes X the number of classes)
	'''
	# bdt = AdaBoostClassifier()
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

	bdt.fit(training_set, training_labels)

	predicted_labels = bdt.predict(testing_set)
	confusion_matrix = metrics.confusion_matrix(testing_labels, predicted_labels)

	print("Confusion matrix:\n%s" % confusion_matrix)
	error = error_measure(predicted_labels, testing_labels)
	print 'error rate %f' % error

	return predicted_labels, confusion_matrix, error

def boosting_B(training_set, training_labels, testing_set, testing_labels):
	'''
	Input Parameters:
		- training_set: a 2-D numpy array that contains training examples (size: the number of training examples X the number of attributes)
		(NOTE: If a training example is 10x10 images, the number of attributes will be 100. You need to reshape your training example)
		- training_labels: a 1-D numpy array that labels of training examples (size: the number of training examples)
		- testing_set: a 2-D numpy array that contains testing examples  (size: the number of testing examples X the number of attributes)
		- testing_labels: a 1-D numpy array that labels of testing examples (size: the number of testing examples)

	Returns:
		- predicted_labels: a 1-D numpy array that contains the labels predicted by the classifier. Labels in this array should be sorted in the same order as testing_labels
		- confusion_matrix: a 2-D numpy array of confusion matrix (size: the number of classes X the number of classes)
	'''
    # Build boosting algorithm for question 6-B
	bdt = AdaBoostClassifier(SVC(probability=True, kernel='linear'), n_estimators=50, learning_rate=1.0, algorithm='SAMME')

	bdt.fit(training_set, training_labels)

	predicted_labels = bdt.predict(testing_set)
	confusion_matrix = metrics.confusion_matrix(testing_labels, predicted_labels)

	print("Confusion matrix:\n%s" % confusion_matrix)
	error = error_measure(predicted_labels, testing_labels)
	print 'error rate %f' % error

	return predicted_labels, confusion_matrix, error

def main():
	"""
	This function runs boosting_A() and boosting_B() for problem 7.
	Load data set and perform adaboosting using boosting_A() and boosting_B()
	"""
	images, labels = load_mnist(digits=range(0, 10), path='.')
	# preprocessing
	images = preprocess(images)

	# pick training and testing set
	# YOU HAVE TO CHANGE THIS TO PICK DIFFERENT SET OF DATA
	erros_1 = []
	erros_2 = []

	for i in xrange(0, 10000, 1000):
		training_set = images[i:i+1000]
		training_labels = labels[i:i+1000]
		testing_set = images[i+1000:i+1100]
		testing_labels = labels[i+1000:i+1100]

		predicted_labels, confusion_matrix, error = boosting_A(training_set, training_labels, testing_set, testing_labels)
		erros_1.append(error)
		predicted_labels, confusion_matrix, error = boosting_B(training_set, training_labels, testing_set, testing_labels)
		erros_2.append(error)

	print erros_1, erros_2


if __name__ == '__main__':
    main()
