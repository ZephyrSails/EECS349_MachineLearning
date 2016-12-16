# Author: Zhiping Xiu
import sys
import numpy as np
import os
import shutil
from cvGen import *
import math
import email
import re
import random
import string

# This function parses the text_file passed into it into a set of words.
# Right now it just splits up the file by blank spaces,
# and returns the set of unique strings used in the file.
def parse(text_file):
	content = text_file.read()

	# following few line is for extracting payload of the email
	# extract payload
	# content = email.message_from_string(content)
	# while content.is_multipart():
	# 	content = content.get_payload()[0]
	# payload = content.get_payload()
	# # preprocess the payload
	# content = re.sub(r'[\W\d_]+', ' ', payload).lower()
	#
	# return set(content.split())

	return set(content.split())	# set is faster for checking exisitance of words


def writedictionary(dictionary, dictionary_filename):
	# Don't edit this function. It writes the dictionary to an output file.
	output = open(dictionary_filename, 'w')
	header = 'word\tP[word|spam]\tP[word|ham]\n'
	output.write(header)
	for k in dictionary:
		line = '{0}\t{1}\t{2}\n'.format(k, str(dictionary[k]['spam']), str(dictionary[k]['ham']))
		output.write(line)


def makeDict(spam, ham, dictionary_filename):
	spam_prior_probability = len(spam)/float((len(spam) + len(ham)))
	words = {}

	for s in spam:
		for word in parse(open(s)):
			if word not in words:
				words[word] = {'spam': 1, 'ham': 0}
			else:
				words[word]['spam'] += 1
	for h in ham:
		for word in parse(open(h)):
			if word not in words:
				words[word] = {'spam': 0, 'ham': 1}
			else:
				words[word]['ham'] += 1

	for word in words:
		words[word]['spam'] = (words[word]['spam']+1.) / float(len(spam)+1)
		words[word]['ham'] = (words[word]['ham']+1.) / float(len(ham)+1)

	writedictionary(words, dictionary_filename)

	return words, spam_prior_probability

	# Following line is solving the 0 prob problem by removing all 0 prob.
	# not a very good way

	# for s in spam:
	# 	for word in parse(open(s)):
	# 		if word not in words:
	# 			words[word] = {'spam': 1, 'ham': 0}
	# 		else:
	# 			words[word]['spam'] += 1
	# for h in ham:
	# 	for word in parse(open(h)):
	# 		if word not in words:
	# 			words[word] = {'spam': 0, 'ham': 1}
	# 		else:
	# 			words[word]['ham'] += 1
	#
	# cleanWords = {}
	# # discard the words that introduce 0 probability
	# for word in words:
	# 	if words[word]['ham'] and words[word]['spam']:
	# 		cleanWords[word] = {'spam': words[word]['spam'] / float(len(spam)),
	# 							'ham': words[word]['ham'] / float(len(ham))}
	#
	# writedictionary(cleanWords, dictionary_filename)
	# return cleanWords, spam_prior_probability


def isSpam(mail, dictionary, spam_prior, priorOnly=0):
	# TODO: Update this function. Right now, all it does is checks whether the
	# spam_prior is more than half the data.
	if priorOnly: return spam_prior >= .5

	# If it is, it says spam for everything. Else, it says ham for everything.
	# You need to update it to make it use the dictionary and the content of
	# the mail. Here is where your naive Bayes classifier goes.
	# spamCount = int(lenDict * spam_prior)
	# hamCount = lenDict - spamCount
	contentSet = parse(open(mail))
	# 				spam 		vs 			ham
	argMax = [math.log(spam_prior), math.log(1-spam_prior)]

	for word in dictionary:
		# extract p
		p_word_t_spam = dictionary[word]['spam']
		# p_word_f_spam = 1 - p_word_t_spam
		p_word_t_ham = dictionary[word]['ham']
		# p_word_f_ham = 1 - p_word_t_ham
		try:
			if word in contentSet:
				argMax[0] += math.log(p_word_t_spam)
				argMax[1] += math.log(p_word_t_ham)
			# else:
				# argMax[0] += math.log(p_word_f_spam)
				# argMax[1] += math.log(p_word_f_ham)
		except:
			print p_word_t_spam, p_word_t_ham # , p_word_f_spam, p_word_f_ham

	return argMax[0] > argMax[1]


def spamSort(mail, spam_directory, ham_directory, dictionary, spam_prior_probability, priorOnly=0):
	positive = 0
	for m in mail:
		# print m
		spam = isSpam(m, dictionary, spam_prior_probability, priorOnly)
		if spam:
			positive += 1
			shutil.copy(m, spam_directory)
		else:
			shutil.copy(m, ham_directory)
	return positive	# use positive to evaluate


# 	Generate a random string tail for dir, prevent the program from reuse old dir.
def genRandomTail(n):
	return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))


# 	cv('sourceDataSet', 'cvSets', 10)
def cv(sourceDir, cvDir, k, priorOnly=0):
	cvDir = cvDir + '_' + genRandomTail(16)
	if not os.path.exists(cvDir): os.mkdir(cvDir)
	cvInputDir = os.path.join(cvDir, 'cvInput')
	if not os.path.exists(cvInputDir): os.mkdir(cvInputDir)
	splitKFold(sourceDir, cvInputDir, k)
	dictDir = os.path.join(cvDir, 'dictDir')
	resultDir = os.path.join(cvDir, 'resultDir')
	if not os.path.exists(dictDir): os.mkdir(dictDir)
	if not os.path.exists(resultDir): os.mkdir(resultDir)

	errorRates = []
	for i, cvDataset in enumerate(cvRead(cvInputDir, k)):
		testingSet, trainingSet = cvDataset
		# create dict for this round (cross validation)
		dictionary, prior = makeDict(trainingSet[1], trainingSet[0], os.path.join(dictDir, 'dict_%d' % i))

		# create result Dir for this round (cross validation)
		resultDirI = os.path.join(resultDir, 'result_%d' % i)
		if not os.path.exists(resultDirI): os.mkdir(resultDirI)
		resultDirISpam = os.path.join(resultDirI, 'spam')
		if not os.path.exists(resultDirISpam): os.mkdir(resultDirISpam)
		resultDirIHam = os.path.join(resultDirI, 'ham')
		if not os.path.exists(resultDirIHam): os.mkdir(resultDirIHam)

		# test positive (spam) testSet
		# print 'cv_%d: test positive (spam) testSet' % i
		true_positive = spamSort(testingSet[1], resultDirISpam, resultDirIHam, dictionary, prior, priorOnly)
		# test negative (ham) testSet
		# print 'cv_%d: test negative (ham) testSet' % i
		false_positive = spamSort(testingSet[0], resultDirISpam, resultDirIHam, dictionary, prior, priorOnly)

		# evaluate
		testingSetCount = len(testingSet[0]) + len(testingSet[1])
		trainingSetCount = len(trainingSet[0]) + len(trainingSet[1])
		errorCount = (len(testingSet[1])-true_positive) + false_positive
		errorRate = float(errorCount) / testingSetCount
		print 'cv_%d: trained on %d emails, tested on %d emails, misclassified: %d emails, errorRate: %f' % (i, trainingSetCount, testingSetCount, errorCount, errorRate)
		errorRates.append(errorRate)

	print 'Mean error rate: %f' % (float(sum(errorRates)) / len(errorRates))
	return errorRates


#
#	use spamSort instead
#
def spamsort(mail_directory, spam_directory, ham_directory, dictionary, spam_prior_probability):

	mail = [os.path.join(mail_directory, f) for f in os.listdir(mail_directory) if os.path.isfile(os.path.join(mail_directory, f))]

	spamSort(mail, spam_directory, ham_directory, dictionary, spam_prior_probability, priorOnly=0)
	# for m in mail:
	# 	content = parse(open(mail_directory + m))
	# 	spam = isSpam(content, dictionary, spam_prior_probability)
	# 	if spam:
	# 		shutil.copy(mail_directory + m, spam_directory)
	# 	else:
	# 		shutil.copy(mail_directory + m, ham_directory)


#
# 	use makeDict instead
#
def makeDictionary(spam_directory, ham_directory, dictionary_filename):
	# Making the dictionary.

	spam = [os.path.join(spam_directory, f) for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]
	ham = [os.path.join(ham_directory, f) for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]

	return makeDict(spam, ham, dictionary_filename)
	# return makeDict(spam, ham, os.path.join(dictDir, 'dict_%d' % i))
	#
	# spam_prior_probability = len(spam)/float((len(spam) + len(ham)))
	#
	# words = {}
	# dictionary, prior = makeDict(spam, ham, os.path.join(dictDir, 'dict_%d' % i))

	# These for loops walk through the files and construct the dictionary.
	# The dictionary, words, is constructed so that words[word]['spam'] gives
	# the probability of observing that word, given we have a spam document
	# P(word|spam), and words[word]['ham'] gives the probability of observing
	# that word, given a hamd document P(word|ham). Right now, all it does is
	# initialize both probabilities to 0.
	# TODO: add code that puts in your estimates for P(word|spam) and
	# P(word|ham).

	# for s in spam:
	# 	for word in parse(open(spam_directory + s)):
	# 		if word not in words:
	# 			words[word] = {'spam': 0, 'ham': 0}
	# for h in ham:
	# 	for word in parse(open(ham_directory + h)):
	# 		if word not in words:
	# 			words[word] = {'spam': 0, 'ham': 0}
	#
	# #Write it to a dictionary output file.
	# writedictionary(words, dictionary_filename)
	#
	# return words, spam_prior_probability


if __name__ == "__main__":
	# Here you can test your functions. Pass it a training_spam_directory,
	# a training_ham_directory, and a mail_directory that is filled with
	# unsorted mail on the command line. It will create two directories in
	# the directory where this file exists: sorted_spam, and sorted_ham.

	# The files will show up in this directories according to the algorithm
	# you developed.

	# Usage: ~ python spamfilter.py 'trainingDir/spam' 'trainingDir/ham' 'testingDir'

	training_spam_directory = sys.argv[1]
	training_ham_directory = sys.argv[2]
	test_mail_directory = sys.argv[3]
	test_spam_directory = 'sorted_spam'
	test_ham_directory = 'sorted_ham'
	if not os.path.exists(test_spam_directory):
		os.mkdir(test_spam_directory)
	if not os.path.exists(test_ham_directory):
		os.mkdir(test_ham_directory)
	dictionary_filename = "dictionary.dict"

	#create the dictionary to be used
	# tinker()
	dictionary, spam_prior_probability = makeDictionary(training_spam_directory, training_ham_directory, dictionary_filename)
	#sort the mail
	spamsort(test_mail_directory, test_spam_directory, test_ham_directory, dictionary, spam_prior_probability)
