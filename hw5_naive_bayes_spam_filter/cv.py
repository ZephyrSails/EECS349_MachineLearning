from spamfilter import *

#
# Usage: ~ python cv.py 'sourceDataSet' 'resultDir' 10
#
# Don't change the first parameter, unless you want to use your own dataset
#
# Every time when you call the function, the program will generate a dir
# with the <resultDir> you given, and a 16 length random string (to prevent reuse dir).
# You may need to clear the generated dir manually.
#
if __name__ == '__main__':
    cv(sys.argv[1], sys.argv[2], int(sys.argv[3]))
