import numpy as np

epsilon = 0.3

articles = dict
values = None

def set_articles(articles):
    print("Articles: ",articles)
    for article in articles:
      #  print("Article: ", article)
        articles[article] = 0
    return articles


def update(reward):
    #print(reward)
    pass


def epsilon_greedy(choices):
    random = np.random.uniform()
    #If random < epsilon, we simply pick a random article
    choice = None
    if random < epsilon:
        article_index = np.random.uniform(0,len(choices))
        choice = choices[article_index]
    else:
        max = -1
        for article in choices:
            if articles[article] > max:
                choice = article
                max = articles[article]

    return choice


def recommend(time, user_features, choices):
    #print("choices: ", choices)
    #choice = epsilon_greedy(choices)
    return choices[0]
