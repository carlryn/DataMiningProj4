from __future__ import print_function
import numpy as np
# from np import *
from numpy.linalg import inv
from numpy import transpose, inner, log, dot
from scipy import linalg
from random import randint


# ALPHA = 1 # unknown parameter
DELTA = 0.05
ALPHA = 1 + (log(2/DELTA)/2)**0.5


# Z = dict()
X = dict()
M = dict()
MInv = dict()

B = dict()

Z = list()

W = dict()

COUNTER = 0 
LastChoice = list()

def set_articles(barticles):
    for key in barticles:
        feature_length = len(barticles[key])
        X[key] = barticles[key] # not used by algorithm
        M[key] = np.eye(feature_length)
        MInv[key] = np.eye(feature_length)

        B[key] = np.zeros(feature_length)

        W[key] = np.zeros(feature_length)


def update(reward):
    global COUNTER
    COUNTER += 1
    print("\rcounter: {}".format(COUNTER), end="")

    if reward == -1:
        return

    if reward == 1:
        r = 0.5
    else:
        r = -20

    x = LastChoice[-1]
    z = Z[-1]

    M[x] += inner(z, z) 
    MInv[x] = linalg.solve(M[x], np.eye(len(z)))
    B[x] += reward * z

    W[x] = dot(MInv[x], B[x])


def recommend(time, user_features, choices):
    UCB = dict()
    z = np.array(user_features)
    Z.append(z)
    for x in choices:
        Minv = MInv[x]
        wx = W[x]
        UCB[x] = inner(wx, z) + \
                 ALPHA * np.sqrt(dot(dot(z, Minv), z))

    maxUCB = max(UCB.values())
    maxChoices = [x for x in UCB if UCB[x] == maxUCB]

    choice = np.random.choice(maxChoices)
    # print(choice)
    LastChoice.append(choice)

    return choice

