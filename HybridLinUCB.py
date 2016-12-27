from __future__ import print_function
import numpy as np
from numpy.linalg import inv
from numpy import transpose, inner, log, dot


DELTA = 0.05
ALPHA = 1 + np.sqrt((log(2/DELTA)/2))
FEATURE_LENGTH = 6
REWARD = 0.5
PUNISH = -21


def mul(x):
    # for a in x:
    #     print(a.shape)
    # print()
    return reduce(lambda x,y: np.dot(x,y), x)

class HybridLinUCB():
    def __init__(self):
        self.COUNTER = 0

        self.X = dict()
        self.B = dict()

        self.A = {0: np.eye(FEATURE_LENGTH)}
        self.AInv = {0: np.eye(FEATURE_LENGTH)}
        self.b = {0: np.zeros([FEATURE_LENGTH, 1])}

        self.LastChoice = None
        self.LastUserFeature = None

    def set_articles(self, articles):
        for k in articles:
            self.X[k] = np.array([articles[k]]).transpose()
            self.A[k] = np.eye(FEATURE_LENGTH)
            self.AInv[k] = np.eye(FEATURE_LENGTH)
            self.B[k] = np.zeros([FEATURE_LENGTH,FEATURE_LENGTH])
            self.b[k] = np.zeros([FEATURE_LENGTH,1])

    def update(self, reward):
        self.COUNTER += 1
        print("\rcounter: {}".format(self.COUNTER), end="")

        if reward == -1:
            return

        if reward == 1:
            r = REWARD
        else:
            r = PUNISH




        a = self.LastChoice
        z = self.LastUserFeature
        x = self.X[a]


        self.A[0] += mul([self.B[a].transpose(), self.AInv[a], self.B[a]])
        self.AInv[0] = inv(self.A[0])
        self.b[0] += mul([self.B[a].transpose(), self.AInv[a], self.b[a]])
        self.A[a] += mul([x, x.transpose()])
        self.AInv[a] = inv(self.A[a])
        self.B[a] += mul([x, z.transpose()])
        self.b[a] += (x * r)
        self.A[0] += mul([z, z.transpose()]) - \
                    mul([self.B[a].transpose(), self.AInv[a], self.B[a]])
        self.b[0] += (r * z - mul([self.B[a].transpose(), self.AInv[a], self.b[a]]))


    def recommend(self, time, user_features, choices):
        def calcP(a, beta, z):
            theta = mul([self.AInv[a], self.b[a] - mul([self.B[a], beta])])
            x = self.X[a]


            sta = mul([z.transpose(), self.AInv[0], z]) - \
                    2 * mul([z.transpose(), self.AInv[0], self.B[a].transpose(), self.AInv[a], x]) + \
                    mul([x.transpose(), self.AInv[a], x]) + \
                    mul([x.transpose(), self.AInv[a], self.B[a], self.AInv[0], self.B[a].transpose(), self.AInv[a], x])
            
            return mul([z.transpose(), beta]) + \
                    mul([x.transpose(), theta]) + \
                    ALPHA * np.sqrt(sta)

        z = np.array([user_features]).transpose()
        beta = mul([self.AInv[0], self.b[0]])

        p = [calcP(k, beta, z) for k in choices]


        choice = choices[np.argmax(p)]

        self.LastChoice = choice
        self.LastUserFeature = z

        return choice


policy = HybridLinUCB()

def set_articles(barticles):
    return policy.set_articles(barticles)

def update(reward):
    return policy.update(reward)

def recommend(time, user_features, choices):
    chosen = policy.recommend(time, user_features, choices)
    print(chosen)
    return chosen

