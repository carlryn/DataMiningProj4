import numpy as np
import stolen_hybrid

hybrid = stolen_hybrid.HybridUCB()


def set_articles(barticles):
    hybrid.set_articles(barticles)

def update(reward):
    hybrid.update(reward)


def recommend(time, user_features, choices):
    choice = hybrid.recommend(time,user_features,choices)

    return choice