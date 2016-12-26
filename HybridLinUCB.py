from __future__ import print_function
import numpy as np
# from np import *
from numpy.linalg import inv
from numpy import transpose, inner, log, dot
from random import randint


DELTA = 0.05
ALPHA = 1 + np.sqrt((log(2/DELTA)/2))

REWARD = 1
PUNISH = -20


A = dict()
AInv = dict()
b = dict()
X = dict()
B = dict()
Z = list()


COUNTER = 0 
LastChoice = list()

def set_articles(barticles):
    feature_length = 0

    for key in barticles:
        feature_length = len(barticles[key])
        X[key] = np.array([barticles[key]]).transpose()
        A[key] = np.eye(feature_length)
        AInv[key] = np.eye(feature_length)
        B[key] = np.zeros((feature_length, feature_length))
        b[key] = np.zeros(feature_length)

    A[0] = np.eye(feature_length)
    AInv[0] = np.eye(feature_length)
    b[0] = np.zeros(feature_length)

    # for key in A:


def update(reward):
    # global COUNTER
    # COUNTER += 1
    # print("\rcounter: {}".format(COUNTER), end="")

    if reward == -1:
        return

    if reward == 1:
        r = REWARD
    else:
        r = PUNISH




    a = LastChoice[-1]
    x = X[a]
    z = Z[-1]


    A[0] += mul([B[a].transpose(), AInv[a], B[a]])
    AInv[0] = inv(A[0])
    b[0] += mul([B[a].transpose(), AInv[a], b[a]])
    A[a] += mul([x, x.transpose()])
    AInv[a] = inv(A[a])
    B[a] += mul([x, z.transpose()])
    b[a] += (x * r)[0]
    A[0] += mul([z, z.transpose()]) - \
            mul([B[a].transpose(), AInv[a], B[a]])
    b[0] += (r * z - mul([B[a].transpose(), AInv[a], b[a]]))[0]


def mul(x):
    return reduce(lambda x,y: np.dot(x,y), x)


def recommend(time, user_features, choices):
    p = dict()
    beta = mul([AInv[0], b[0]])
    z = np.array([user_features]).transpose()
    Z.append(z)
    for a in choices:
        x = X[a]
        theta = mul([AInv[a], b[a] - mul([B[a], beta])])
        sta = mul([z.transpose(), AInv[0], z]) - \
              2 * mul([z.transpose(), AInv[0], B[a].transpose(), AInv[a], x]) + \
              mul([x.transpose(), AInv[a], x]) + \
              mul([x.transpose(), AInv[a], B[a], AInv[0], B[a].transpose(), AInv[a], x])
        p[a] = mul([z.transpose(), beta]) + \
               mul([x.transpose(), theta]) + \
               ALPHA * np.sqrt(sta)

    maxpa = max(p.values())
    maxChoices = [x for x in p if p[x] == maxpa]
    choice = np.random.choice(maxChoices)
    LastChoice.append(choice)

    return choice

