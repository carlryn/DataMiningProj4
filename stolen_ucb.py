import numpy as np
import math
from scipy import linalg
from numpy.linalg import inv




# upper bound coefficient
alpha = 3  # if worse -> 2.9, 2.8 1 + np.sqrt(np.log(2/delta)/2)
r1 = 0.5  # if worse -> 0.7, 0.8
r0 = -20  # if worse, -19, -21
# dimension of user features = d
d = 6
# Aa : collection of matrix to compute disjoint part for each article a, d*d
Aa = {}
# AaI : store the inverse of all Aa matrix
AaI = {}
# ba : collection of vectors to compute disjoin part, d*1
ba = {}

a_max = 0

theta = {}
x = None
xT = None
# linUCB

def set_articles(art):
    # init collection of matrix/vector Aa, Ba, ba
    for key in art:
        #Aa[key] = np.identity(d)
        Aa[key] = np.diag(art[key])
        ba[key] = np.zeros((d, 1))
        AaI[key] = inv(Aa[key])
        theta[key] = np.zeros((d, 1))

def update(reward):
    if reward == -1:
        pass
    elif reward == 1 or reward == 0:
        if reward == 1:
            r = r1
        else:
            r = r0
        Aa[a_max] += x.dot(xT)
        ba[a_max] += r * x
        AaI[a_max] = linalg.solve(Aa[a_max], np.identity(d))
        theta[a_max] = AaI[a_max].dot(ba[a_max])
    else:
        # error
        pass

def recommend(time, user_features, articles):

    xaT = np.array([user_features])
    xa = np.transpose(xaT)
    # art_max = -1
    # old_pa = 0
    pa = np.array(
        [float(np.dot(xaT, theta[article]) + alpha * np.sqrt(np.dot(xaT.dot(AaI[article]), xa))) for
         article in articles])
    global a_max
    a_max = articles[divmod(pa.argmax(), pa.shape[0])[1]]
    '''
    for article in articles:
        # x : feature of current article, d*1
        # theta = self.AaI[article].dot(self.ba[article])
        sa = np.dot(xaT.dot(self.AaI[article]), xa)
        new_pa = float(np.dot(xaT, self.theta[article]) + self.alpha * np.sqrt(np.dot(xaT.dot(self.AaI[article]), xa)))
        if art_max == -1:
            old_pa = new_pa
            art_max = article
        else:
            if old_pa < new_pa:
                art_max = article
                old_pa = new_pa
    '''
    global x
    x = xa
    global xT
    xT = xaT
    # article index with largest UCB
    # self.a_max = art_max # divmod(pa.argmax(), pa.shape[0])[1]
    print(a_max)
    return a_max