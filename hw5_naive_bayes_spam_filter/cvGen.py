import sys
import numpy as np
import os
import shutil
import random

# 'tinker,'.strip(r'\W')

#   splitKFold('sourceDataSet', 'crossValidation', 5)
#   Generate k fold cross validation dataset for a group of dataset files.
#   input example:
#       /sourceDir:
#           /easy_ham
#           /spam
#   output example:
#       /targetDir
#           /easy_ham
#               /0
#               /1..
#               /k-1
#           /spam
#               /0
#               /1..
#               /k-1
def splitKFold(sourceDir, targetDir, k):
    sourceDirs = [d for d in os.listdir(sourceDir)
                    if os.path.isdir(os.path.join(sourceDir, d))]

    if not os.path.exists(targetDir):
		os.mkdir(targetDir)

    for d in sourceDirs:
        tempSourceDir = os.path.join(sourceDir, d)
        tempTargetDir = os.path.join(targetDir, d)
        sourceDatas = [f for f in os.listdir(tempSourceDir) if os.path.isfile(os.path.join(tempSourceDir, f))]
        if not os.path.exists(tempTargetDir):
    		os.mkdir(tempTargetDir)
        random.shuffle(sourceDatas)

        for idx, sdir in enumerate(chunks(sourceDatas, k)):
            groupDir = os.path.join(tempTargetDir, str(idx))
            if not os.path.exists(groupDir):
                os.mkdir(groupDir)
            for f in sdir:
                shutil.copy(os.path.join(tempSourceDir, f), groupDir)


# evenly divid a list to n part
def chunks(lst, n):
    l = [(float(len(lst))/n) * i for i in xrange(n+1)]
    for f, t in zip(l, l[1:]):
        yield lst[int(f):int(t)]


#   yield generated file in, for the test.
def cvRead(cvDir, k):
    cvDirs = [os.path.join(cvDir, d) for d in os.listdir(cvDir)
                    if os.path.isdir(os.path.join(cvDir, d))]

    for i in xrange(k):
        testingSet = []
        trainingSet = []
        for d in cvDirs:
            trainingGroup = []
            groupDir = os.path.join(d, str(i))
            testingSet.append([os.path.join(groupDir, f) for f in os.listdir(groupDir) if os.path.isfile(os.path.join(groupDir,f)) and f[0] != '.'])
            for j in xrange(k):
                if j != i:
                    groupDir = os.path.join(d, str(j))
                    trainingGroup += [os.path.join(groupDir, f) for f in os.listdir(groupDir) if os.path.isfile(os.path.join(groupDir,f)) and f[0] != '.']

            trainingSet.append(trainingGroup)
        yield testingSet, trainingSet


if __name__ == '__main__':
    # splitKFold('sourceDataSet', 'cvSets', 5)
    print [(i[0], len(j[1])) for i, j in cvRead('cvSets', 5)]
