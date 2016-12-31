from __future__ import print_function
import numpy as np
# from np import *
from numpy.linalg import inv
from numpy import transpose, inner, log, dot
from random import randint


DELTA = 0.001
ALPHA = 1 + np.sqrt((log(2/DELTA)/2))


REWARD = 100
PUNISH = -20

FEATURE_LENGTH = 6 # hard coded because i'm too lazy to collect this number on runtime

def mul(x):
    return reduce(lambda x,y: np.dot(x,y), x)

class LinUCB():
    def __init__(self):
        self.COUNTER = 0 
        self.LastChoice = None
        self.LastUserFeature = None

    def set_articles(self, barticles):
        self.X = {k: v for k,v in barticles.items()} # Not used
        self.M = {k: np.eye(FEATURE_LENGTH,FEATURE_LENGTH) for k,v in barticles.items()}
        self.MInv = {k: inv(v) for k,v in self.M.items()}
        self.B = {k: np.zeros([FEATURE_LENGTH, 1]) for k in barticles.keys()}


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

        self.M[x] += mul([z, z.transpose()]) 
        self.MInv[x] = inv(self.M[x])
        self.B[x] += r * z


    def recommend(self, time, user_features, choices):
        def CalcUCB(a, z):
            wx = mul([self.MInv[a], self.B[a]])

            return mul([wx.transpose(), z]) + \
                    ALPHA * np.sqrt(mul([z.transpose(), self.MInv[a], z]))

        user_features[0] = time
        x = np.array([user_features]).transpose()
        UCB = [CalcUCB(k, x) for k in choices]

        choice = choices[np.argmax(UCB)]

        self.LastChoice = choice
        self.LastUserFeature = x

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



