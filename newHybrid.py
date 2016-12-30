from __future__ import print_function

import numpy as np
from numpy import log, sqrt
from numpy.linalg import inv
import sys

DELTA = 0.001
ALPHA = 1 + np.sqrt((log(2/DELTA)/2))
USER_FEATURE_LEN = 5 #k
ART_FEATURE_LEN  = 6 #d
REWARD = 0.5
PUNISH = -20

def mul(x):
    return reduce(lambda x,y: np.dot(x,y), x)

class HybridUCB:

    def __init__(self):

        self.A0 = np.eye(ART_FEATURE_LEN)
        self.A0Inv = np.eye(ART_FEATURE_LEN)
        self.b0 = np.zeros([ART_FEATURE_LEN, 1])


        self.A = dict()
        self.AInv = dict()
        self.B = dict()
        self.b = dict()

        self.COUNTER = 0

    def set_articles(self, art):
        self.Art = {k:np.array([v]).transpose() for k,v in art.items()}




    def update(self, reward):
        self.COUNTER += 1
        print("\rcounter: {}".format(self.COUNTER), end="")

        if reward == -1:
            return

        if reward == 1:
            r = REWARD
        else:
            r = PUNISH

        x = self.LastUserF
        z = self.Art[self.LastChoice]
        a = tuple(z.flatten()) + tuple(x.flatten())

        self.A0 += mul([self.B[a].transpose(), self.AInv[a], self.B[a]])
        # self.A0Inv no need to update because it is not used before next update
        self.b0 += mul([self.B[a].transpose(), self.AInv[a], self.b[a]])
        self.A[a] += mul([x, x.transpose()])
        self.AInv[a] = inv(self.A[a])
        self.B[a] += mul([x, z.transpose()])
        self.b[a] += r * x
        self.A0 += mul([z, z.transpose()]) - mul([self.B[a].transpose(), self.AInv[a], self.B[a]])
        self.A0Inv = inv(self.A0)
        self.b0 += r * z - mul([self.B[a].transpose(), self.AInv[a], self.b[a]])


    def recommend(self, timestamp, user_features, articles):
        x = np.array([user_features[1:]]).transpose()
        p = dict()

        beta = mul([self.A0Inv, self.b0])
        for art in articles:
            z = self.Art[art]

            a = tuple(z.flatten()) + tuple(x.flatten())

            if a not in self.A:
                self.A[a] = np.eye(USER_FEATURE_LEN)
                self.AInv[a] = np.eye(USER_FEATURE_LEN)
                self.B[a] = np.zeros([USER_FEATURE_LEN, ART_FEATURE_LEN])
                self.b[a] = np.zeros([USER_FEATURE_LEN, 1])


            theta = mul([self.AInv[a], self.b[a] - mul([self.B[a], beta])])

            sta = mul([z.transpose(), self.A0Inv, z]) - \
                  2 * mul([z.transpose(), self.A0Inv, self.B[a].transpose(), self.AInv[a], x]) + \
                  mul([x.transpose(), self.AInv[a], x]) + \
                  mul([x.transpose(), self.AInv[a], self.B[a], self.A0Inv, self.B[a].transpose(), self.AInv[a], x])

            p[art] = mul([z.transpose(), beta]) + mul([x.transpose(), theta]) + ALPHA * sqrt(sta)


        maxp = max(p.values())
        maxarts = [art for art in articles if p[art] == maxp]
        choice = np.random.choice(maxarts)
        self.LastChoice = choice
        self.LastUserF = x
        # print(choice)
        return choice




policy = HybridUCB()

def set_articles(barticles):
    return policy.set_articles(barticles)

def update(reward):
    return policy.update(reward)

def recommend(time, user_features, choices):
    chosen = policy.recommend(time, user_features, choices)
    # print(chosen)
    return chosen
