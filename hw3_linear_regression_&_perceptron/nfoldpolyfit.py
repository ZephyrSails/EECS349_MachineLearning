#	Starter code for linear regression problem
#	Below are all the modules that you'll need to have working to complete this problem
#	Some helpful functions: np.polyfit, scipy.polyval, zip, np.random.shuffle, np.argmin, np.sum, plt.boxplot, plt.subplot, plt.figure, plt.title
import sys
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#	NFOLDPOLYFIT Fit polynomial of the best degree to data.
#   NFOLDPOLYFIT(X,Y,maxDegree, nFold, verbose) finds and returns the coefficients
#   of a polynomial P(X) of a degree between 1 and N that fits the data Y
#   best in a least-squares sense, averaged over nFold trials of cross validation.
#
#   P is a vector (in numpy) of length N+1 containing the polynomial coefficients in
#   descending powers, P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1). use
#   numpy.polyval(P,Z) for some vector of input Z to see the output.
#
#   X and Y are vectors of datapoints specifying input (X) and output (Y)
#   of the function to be learned. Class support for inputs X,Y:
#   float, double, single
#
#   maxDegree is the highest degree polynomial to be tried. For example, if
#   maxDegree = 3, then polynomials of degree 0, 1, 2, 3 would be tried.
#
#   nFold sets the number of folds in nfold cross validation when finding
#   the best polynomial. Data is split into n parts and the polynomial is run n
#   times for each degree: testing on 1/n data points and training on the
#   rest.
#
#   verbose, if set to 1 shows mean squared error as a function of the
#   degrees of the polynomial on one plot, and displays the fit of the best
#   polynomial to the data in a second plot.
#
#
#   AUTHOR: Zhiping Xiu
#
#	USAGE:
# 	python nfoldpolyfit.py <csvfile> <maxdegree> <numberoffolds> <verbose>
# 	python nfoldpolyfit.py 'linearreg.csv' '3' '1' '0'
def nfoldpolyfit(X, Y, maxK, n, verbose):
	# nfoldpoly(X, Y, maxK, n, verbose)
	rss_record = [nfoldpoly(X, Y, k, n) for k in range(maxK+1)]
	rss_record = [sum([nfoldpoly(X, Y, k, n) for i in range(10)])/10 for k in range(maxK+1)]

	for k in range(len(rss_record)):
		print "Averaged Mean Square Error for degree %d %d-fold cross validation is: %f" % (k, n, rss_record[k])

	best_rss_degree = rss_record.index(min(rss_record))
	print "Degree %d is the best, which has averaged Mean Square Error of: %f" % (best_rss_degree, rss_record[best_rss_degree])

	if verbose:
		plot_rss_record(rss_record)
		nfoldpoly(X, Y, best_rss_degree, n, True)

	# print "please notice:\n\tAveraged Mean Square Error means I get 10 %d-fold cross validation\n\t\tfor each degree, and get an average\n\tIt doesn't means I don't know the meaning of RSS or Cross Validation" % (n)



def plot_rss_record(rss_record):
	plt.plot(range(len(rss_record)), rss_record)
	for i in range(len(rss_record)):
		plt.text(i, rss_record[i], str(round(rss_record[i], 3)))
	plt.title("Mean Square Error up to %d degree" % (len(rss_record)-1))
	plt.xlim([-1, len(rss_record)])

	plt.xticks(range(len(rss_record)))
	plt.show()

def nfoldpoly(X, Y, K, n, verbose=False):
	Xs = get_poly_X(X, K)

	if n > 1:
		spliter = [int(i*(1.*len(X)/n)) for i in range(n)] + [len(X)]

		X_Y = zip(X, Y)
		np.random.shuffle(X_Y)
		sum_rss = 0
		for i in range(n):
			training_X, training_Y = zip(*(X_Y[:spliter[i]] + X_Y[spliter[i+1]:]))

			testing_X, testing_Y = zip(*X_Y[spliter[i]:spliter[i+1]])

			poly_training_X = get_poly_X(training_X, K)
			W = get_W(poly_training_X, training_Y)

			hypo_Y = hypo(testing_X, W)
			rss = get_RSS(testing_Y, hypo_Y)
			sum_rss += rss
			# print "%d degree %d-fold cross validation, test %d/%d-RSS: %f" % (K, n, i+1, n, rss)
			# plot(training_X, training_Y, W, "%d degree %d-fold Cross Validation %d/%d-RSS: %f" % (K, n, i+1, n, rss), testing_X, testing_Y)
		# print "%d degree %d-fold cross validation end, Average RSS: %f" % (K, n, sum_rss/n)

	W = get_W(Xs, Y)
	hypo_Y = hypo(X, W)
	if verbose:
		# plot(X, Y, W, "%d degree, Using full input for-RSS: %f" % (K, get_RSS(Y, hypo_Y)))
		print "Print chart: %d degree, Using full input" % (K)
		plot(X, Y, W, "%d degree, Using full input" % (K))
		# plot(training_X, training_Y, W, "Predict 3.0", np.array([3]), np.array([hypo(np.array([3]), W)]))
		# print "%d degree, Using full input-RSS: %f" % (K, get_RSS(Y, hypo_Y))

	return sum_rss/n

def get_RSS(testing_Y, hypo_Y):
	return sum(abs(testing_Y - hypo_Y)**2)

def hypo(X, W):
	return np.array([get_y(x, W) for x in X])

def get_y(x, W):
	return np.dot(np.array([x**n for n in range(len(W))]).T, W)

def split_n_fold(n, X, Y):
	np.random.shuffle(zip(X, Y))

def get_poly_X(X, K):
	return np.array([[x**n for n in range(K+1)] for x in X])

def get_W(X, Y):
	return np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))

def plot(X, Y, W, title, t_X=[], t_Y=[]):
	xs = np.arange(-1.5, 4, 0.05)

	if t_X:
		plt.scatter(t_X, t_Y, s=81, color="red", alpha=0.5, label='Prediction Set')
		plt.text(t_X[0], t_Y[0], str(round(t_Y[0][0], 3)))
		plt.xlim([-1.5, 4])
	else:

		plt.xlim([-1.5, 1.5])
		plt.ylim([-1.5, 3])

	ys = hypo(xs, W)

	plt.plot(xs, ys, color="blue", label="Regression Line")
	plt.scatter(X, Y, s=81, color="blue", alpha=0.5, label='Training Set')


	plt.legend()
	plt.title(title)

	plt.show()

def main():
	# read in system arguments, first the csv file, max degree fit, number of folds, verbose
	rfile = sys.argv[1]
	maxK = int(sys.argv[2])
	nFolds = int(sys.argv[3])
	# verbose = bool(sys.argv[4])
	verbose = True if sys.argv[4] == "1" else False

	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X = []
	Y = []
	# put the x coordinates in the list X, the y coordinates in the list Y
	for i, row in enumerate(dat):
		if i > 0:
			X.append(float(row[0]))
			Y.append(float(row[1]))
	X = np.array(X)
	Y = np.array(Y)
	nfoldpolyfit(X, Y, maxK, nFolds, verbose)

if __name__ == "__main__":
	main()
