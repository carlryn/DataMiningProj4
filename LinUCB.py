from __future__ import print_function
import numpy as np
# from np import *
from numpy.linalg import inv
from numpy import transpose, inner, log, dot
from random import randint
import os

DELTA = 0.001
ALPHA = 1 + np.sqrt((log(2/DELTA)/2))

REWARD = float(os.environ['REWARD'])
PUNISH = float(os.environ['PUNISH'])

FEATURE_LENGTH = 6 # hard coded because i'm too lazy to collect this number on runtime

class LinUCB():
    def __init__(self):
        self.COUNTER = 0 
        self.LastChoice = None
        self.LastUserFeature = None

    def set_articles(self, barticles):
        self.X = {k: v for k,v in barticles.items()}
        self.M = {k: np.identity(FEATURE_LENGTH) for k,v in barticles.items()}
        self.MInv = {k: inv(v) for k,v in self.M.items()}
        self.B = {k: np.zeros(FEATURE_LENGTH) for k in barticles.keys()}
        self.W = {k: np.zeros(FEATURE_LENGTH) for k in barticles.keys()}


    def update(self, reward):
        self.COUNTER += 1
        #print("\rcounter: {}".format(self.COUNTER), end="")

        if reward == -1:
            return

        r = REWARD if reward == 1 else PUNISH

        x = self.LastChoice
        z = self.LastUserFeature

        self.M[x] += np.outer(z, z) 
        self.MInv[x] = inv(self.M[x])

        self.B[x] += dot(r, z)

        self.W[x] = dot(self.MInv[x], self.B[x])


    def recommend(self, time, user_features, choices):
        z = np.log(np.array(user_features)+2)
        idx = np.argmax(map(lambda x: dot(self.W[x], z) + ALPHA * np.sqrt(dot(dot(z, self.MInv[x]), z)), choices))
        choice = choices[idx]

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
    #print(chosen)
    return chosen



