from __future__ import print_function
import numpy as np
# from np import *
from numpy.linalg import inv
from numpy import transpose, inner, log, dot
from random import randint


DELTA = 0.001
ALPHA = 1 + np.sqrt((log(2/DELTA)/2))


REWARD = 0.5
PUNISH = -20

FEATURE_LENGTH = 6 # hard coded because i'm too lazy to collect this number on runtime

class LinUCB():
    def __init__(self):
        self.COUNTER = 0 
        self.LastChoice = None
        self.LastUserFeature = None

    def set_articles(self, barticles):
        self.X = {k: v for k,v in barticles.items()}
        self.M = {k: np.eye(FEATURE_LENGTH,FEATURE_LENGTH) for k,v in barticles.items()}
        self.MInv = {k: inv(v) for k,v in self.M.items()}
        self.B = {k: np.zeros(FEATURE_LENGTH) for k in barticles.keys()}
        self.W = {k: np.zeros(FEATURE_LENGTH) for k in barticles.keys()}


    def update(self, reward):
        self.COUNTER += 1
        print("\rcounter: {}".format(self.COUNTER), end="")

        if reward == -1:
            return

        if reward == 1:
            r = REWARD
        else:
            r = PUNISH

        x = self.LastChoice
        z = self.LastUserFeature

        self.M[x] += np.outer(z, z) 
        self.MInv[x] = inv(self.M[x])

        self.B[x] += r * z[:,0]

        self.W[x] = dot(self.MInv[x], self.B[x])


    def recommend(self, time, user_features, choices):
        def mul(x):
            return reduce(lambda x,y: np.dot(x,y), x)
        def CalcUCB(x, z):
            b = self.B[x]
            Minv = self.MInv[x]
            wx = mul([Minv, b])

            return float(np.inner(wx, z[:,0])) + \
                    ALPHA * np.sqrt(mul([z.transpose(), Minv, z]))

        z = np.array([user_features]).transpose()
        UCB = [CalcUCB(k, z) for k in choices]

        # maxUCB = max(UCB.values())
        # maxChoices = [x for x in UCB if UCB[x] == maxUCB]

        # choice = np.random.choice(maxChoices)
        choice = choices[np.argmax(UCB)]

        self.LastChoice = choice
        self.LastUserFeature = z

        return choice


policy = LinUCB()

def set_articles(barticles):
    return policy.set_articles(barticles)

def update(reward):
    return policy.update(reward)

def recommend(time, user_features, choices):
    chosen = policy.recommend(time, user_features, choices)
    # print(chosen)
    return chosen



