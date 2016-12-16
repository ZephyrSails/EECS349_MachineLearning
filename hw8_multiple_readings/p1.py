import random

def mutate(B, p):
    for i in xrange(len(B)):
        if random.random() < p:
            B[i] = not B[i]

def acc(A, B):
    return float(sum([1 for i in xrange(len(A)) if A[i] == B[i]])) / len(A)

if __name__ == '__main__':
    A = [True for _ in xrange(100000)]
    B = [True for _ in xrange(100000)]
    p = 0.2

    a = 1.0
    for _ in xrange(100):
        a = a * (1-p) + (1-a) * p
        mutate(B, p)
        print acc(A, B), a
