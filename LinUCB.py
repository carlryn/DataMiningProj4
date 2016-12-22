from __future__ import print_function
import numpy as np
# from np import *
from numpy.linalg import inv
from numpy import transpose, inner, log

from random import randint


# ALPHA = 1 # unknown parameter
DELTA = 0.05
ALPHA = 1 + (log(2/DELTA)/2)**0.5


# Z = dict()
X = dict()
M = dict()
B = dict()

Z = list()

COUNTER = 0 

def set_articles(barticles):
    X = {key: np.asarray(barticles[key]) for key in barticles} # not used by algorithm


def update(reward):
    global COUNTER
    COUNTER += 1
    print("\rcounter: {}".format(COUNTER), end="")
    z = Z[-1]
    for x in M:
        M[x] += inner(z, z)
        B[x] += reward * z



def recommend(time, user_features, choices):
    UCB = dict()
    z = np.array(user_features)
    Z.append(z)
    for x in choices:
        if x not in M:
            M[x] = np.eye(len(user_features))
            B[x] = np.zeros(len(user_features))
        wx = inner(inv(M[x]), B[x])
        UCB[x] = inner(wx, z) + 
                 ALPHA * inner(inner(z, inv(M[x])), z) ** 0.5

    l = [(UCB[x], x) for x in UCB]

    return np.sort(l)[-1][1]

