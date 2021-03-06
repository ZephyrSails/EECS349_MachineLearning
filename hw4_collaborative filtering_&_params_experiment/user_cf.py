# Starter code for uesr-based collaborative filtering
# Complete the function user_based_cf below. Do not change it arguments and return variables.
# Do not change main() function,

# import modules you need here.
import sys
# import scipy.stats
# import scipy.spatial.distance
import numpy as np
from loader import *
from helper import *


def user_based_cf(datafile, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems):
    '''
    build user-based collaborative filter that predicts the rating
    of a user for a movie.
    This function returns the predicted rating and its actual rating.

    Parameters
    ----------
    <datafile> - a fully specified path to a file formatted like the MovieLens100K data file u.data
    <userid> - a userId in the MovieLens100K data
    <movieid> - a movieID in the MovieLens 100K data set
    <distance> - a Boolean. If set to 0, use Pearson's correlation as the distance measure. If 1, use Manhattan distance.
    <k> - The number of nearest neighbors to consider
    <iFlag> - A Boolean value. If set to 0 for user-based collaborative filtering,
    only users that have actual (ie non-0) ratings for the movie are considered in your top K.
    For user-based, use only movies that have actual ratings by the user in your top K.
    If set to 1, simply use the top K regardless of whether the top K contain actual or filled-in ratings.

    returns
    -------
    trueRating: <userid>'s actual rating for <movieid>
    predictedRating: <userid>'s rating predicted by collaborative filter for <movieid>

    AUTHOR: Zhiping Xiu
    '''
    # Please Check out:
    #     manhattanDistance(), pearsonrDistance() and kNNRating()
    # at
    #     helper.py
    #
    # because those three functions are identical for both item-based and user-based KNN
    matrix = read(datafile)
    predictedRating = kNNRating(userid, movieid, matrix, distance, k, iFlag)
    trueRating = matrix[userid][movieid]
    return trueRating, predictedRating


#   Usage Guide:
#   python user_cf.py 'ml-100k/u.data' 196 242 0 10 0
def main():
    datafile = sys.argv[1]
    userid = int(sys.argv[2])
    movieid = int(sys.argv[3])
    distance = int(sys.argv[4])
    k = int(sys.argv[5])
    i = int(sys.argv[6])
    numOfUsers = 943
    numOfItems = 1682

    trueRating, predictedRating = user_based_cf(datafile, userid, movieid, distance, k, i, numOfUsers, numOfItems)
    print 'userID:{} movieID:{} trueRating:{} predictedRating:{} distance:{} K:{} I:{}'\
    .format(userid, movieid, trueRating, predictedRating, distance, k, i)


if __name__ == "__main__":
    main()
