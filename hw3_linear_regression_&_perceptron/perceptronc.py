import sys
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt


#	This is built apon perceptrona.py
#	A little change is applied to fit n-order polynomial fit.
#	But we are still assuming the input data is 1-order list, with two class output.
#	We are just transform it to n-order polynomial for data that's not linearly saperable.
#
#   X and Y are vectors of datapoints specifying input (X) and output (Y)
#   of the function to be learned.
#	In this case, X single facter input,
#	The output of Y is just showing the classfication, not showing the actual value
#	In another word, you can consider this whole process is performed in a 1-D space.
#
#	w is the weight vector, which is generated with all zero list.
# 	Since we are not allowed to change the input style.
# 	You can only change the polynomial order by changing the w_init, manually.
# 	E.g.
#		We are performing 2-order polynomial fit,
#		so:
#			w_init = np.array([0, 0, 0])
#		if you want to see 3-order polynomial fit,
# 		change it to:
#			w_init = np.array([0, 0, 0, 0])
#
#   AUTHOR: Zhiping Xiu
#
# 	Usage sample:
#	python perceptronc.py "linearclass.csv"
def perceptrona(w, X, Y):
	i = 0
	k = 0
	correct_count = 0		# Used to see if the classfication is all correct
	preped_X = prep_X(X, len(w)-1)	# prep_X for polynomial extensibility, not used for now
	while True:
		k += 1
		i = (i+1) % len(X)
		x = preped_X[i]
		y = Y[i]
		if get_y(x, w) == y:
			correct_count += 1
		else:
			w = w + x * y
			correct_count = 0
		if correct_count == len(X): 	# all correct, jobs done
			print "the final weights is %s, founded on %dth iterations" % (str(w), k)
			plot(X, Y, w)
			# figure out (w, k) and return them here. w is the vector of weights, k is how many iterations it took to converge.
			return (w, k)


#	Get prediction answer with input
def get_y(x, w):
	return 1 if np.dot(x, w) > 0 else -1


# 	Plot the scatter graph with saperation line
def plot(X, Y, w):
	colors = ["blue" if y == 1 else "red" for y in Y]

	for root in np.roots(w[::-1]):
		plt.plot([root,  root], [-1, 1],color="purple", label="Final Saperation Line")

	plt.scatter(X, np.array([0 for i in range(len(X))]), c=colors, s=81, alpha=0.5)
	plt.legend()
	plt.title("Perceptron Saperation")
	# plt.xlim([-5, 10])
	plt.show()


#	prep_X for polynomial extensibility
# 	Used to add dimension according to the input data
# 	add polynomial degree if dataset is not linearly saperable.
def prep_X(X, k=1):
	return np.array([[x**a for a in range(k+1)] for x in X])


# python perceptrona.py "linearclass.csv"
def main():
	rfile = sys.argv[1]

	#read in csv file into np.arrays X1, X2, Y1, Y2
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X1 = []
	Y1 = []
	X2 = []
	Y2 = []
	for i, row in enumerate(dat):
		if i > 0:
			X1.append(float(row[0]))
			X2.append(float(row[1]))
			Y1.append(float(row[2]))
			Y2.append(float(row[3]))
	X1 = np.array(X1)
	X2 = np.array(X2)
	Y1 = np.array(Y1)
	Y2 = np.array(Y2)

	w_init = np.array([0, 0, 0])

	# perceptrona(w_init, X1, Y1)	# Un-comment this line to see dead loop for a linearly saperable input, it would still work.
	perceptrona(w_init, X2, Y2)


if __name__ == "__main__":
	main()
