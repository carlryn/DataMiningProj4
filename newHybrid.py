from __future__ import print_function

import numpy as np
from numpy import log, sqrt
from numpy.linalg import inv
import sys

DELTA = 0.001
ALPHA = 1 + np.sqrt((log(2/DELTA)/2))
USER_FEATURE_LEN = 6 #k
ART_FEATURE_LEN  = 6 #d
REWARD = 0.5
PUNISH = -20

def mul(x):
    return reduce(lambda x,y: np.dot(x,y), x)

class HybridUCB:

    def __init__(self):

        self.A0 = np.eye(USER_FEATURE_LEN)
        self.A0Inv = np.eye(USER_FEATURE_LEN)
        self.b0 = np.zeros([USER_FEATURE_LEN, 1])


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

        z = self.LastUserF
        x = self.Art[self.LastChoice]
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


        beta = mul([self.A0Inv, self.b0])
        z = np.array([user_features]).transpose()

        p = dict()

        ztA0Inv = mul([z.transpose(), self.A0Inv])
        ztA0Iz = mul([ztA0Inv, z])

        for art in articles:
            a = tuple(user_features) + tuple(self.Art[art].flatten())

            if a not in self.A:
                self.A[a] = np.eye(ART_FEATURE_LEN)
                self.AInv[a] = np.eye(ART_FEATURE_LEN)
                self.B[a] = np.zeros([ART_FEATURE_LEN, USER_FEATURE_LEN])
                self.b[a] = np.zeros([ART_FEATURE_LEN, 1])

            x = self.Art[art]

            theta = mul([self.AInv[a], self.b[a] - mul([self.B[a], beta])])

            sta = ztA0Iz - \
                  2 * mul([z.transpose(), self.A0Inv, self.B[a].transpose(), self.AInv[a], x]) + \
                  mul([x.transpose(), self.AInv[a].transpose(), x]) + \
                  mul([x.transpose(), self.AInv[a], self.B[a], self.A0Inv, self.B[a].transpose(), self.AInv[a], x])

            p[art] = mul([z.transpose(), beta]) + mul([x.transpose(), theta]) + ALPHA * sqrt(sta)


        maxp = max(p.values())
        maxarts = [art for art in articles if p[art] == maxp]
        choice = np.random.choice(maxarts)
        self.LastChoice = choice
        self.LastUserF = z
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
