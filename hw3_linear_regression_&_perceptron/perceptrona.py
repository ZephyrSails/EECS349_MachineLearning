import sys
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt


#   X and Y are vectors of datapoints specifying input (X) and output (Y)
#   of the function to be learned.
#	In this case, X single facter input,
#	The output of Y is just showing the classfication, not showing the actual value
#	In another word, you can consider this whole process is performed in a 1-D space.
#
#	w is the weight vector, which is generated with all zero list.
#
#   AUTHOR: Zhiping Xiu
#
# 	Usage sample:
#	python perceptrona.py "linearclass.csv"
def perceptrona(w, X, Y):
	i = 0
	k = 0
	correct_count = 0		# Used to see if the classfication is all correct
	preped_X = prep_X(X)	# prep_X for polynomial extensibility, not used for now
	while True:
		k += 1
		i = (i+1) % len(X)
		x = preped_X[i]
		y = Y[i]
		if get_y(x, w) == y:
			correct_count += 1
		else:
			w = w + x * y
			# plot_process(X, w, correct_count) # Un-comment this line to track the changes of the saperation line.
			correct_count = 0
		if correct_count == len(X): 	# all correct, jobs done
			print "the final weights is %s, founded on %dth iterations" % (str(w), k)
			plot(X, Y, w)
			# figure out (w, k) and return them here. w is the vector of weights, k is how many iterations it took to converge.
			return (w, k)



#	Get prediction answer with input
def get_y(x, w):
	return 1 if np.dot(x, w) > 0 else -1


# 	This is used to track the saperation line changes in each iterations, not shown in final submission.
# 	If you want to see this, uncomment line 34
def plot_process(X, w, correct_count):
	if w[1] != 0:
		plt.plot([-1.0*w[0]/w[1],  -1.0*w[0]/w[1]], [-1, 1], alpha=(1. * correct_count/len(X)), color="red")


# 	Plot the scatter graph with saperation line
def plot(X, Y, w):
	colors = ["blue" if y == 1 else "red" for y in Y]
	plt.plot([-1.0*w[0]/w[1],  -1.0*w[0]/w[1]], [-1, 1],color="purple", label="Final Saperation Line")
	plt.scatter(X, np.array([0 for i in range(len(X))]), c=colors, s=81, alpha=0.5)
	plt.legend()
	plt.title("Perceptron Saperation")
	plt.xlim([-5, 10])
	plt.show()


# def hypo(X, W):
# 	return np.array([get_y(x, W) for x in X])
# def get_y(x, W):
# 	return np.dot(np.array([x**n for n in range(len(W))]).T, W)


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

	w_init = np.array([0, 0])	# INTIALIZE W_INIT

	perceptrona(w_init, X1, Y1)
	# perceptrona(w_init, X2, Y2)	# Un-comment this line to see dead loop for a non linearly saperable input


if __name__ == "__main__":
	main()
