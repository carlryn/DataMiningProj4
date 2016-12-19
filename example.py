import numpy as np

epsilon = 0.3

articles = dict()
articles_prob = dict()

def set_articles(barticles):
    for key in barticles.keys():
        articles[key] = barticles[key]
        articles_prob[key] = 0


def update(reward):
    print(articles_prob[last_choice])
    print(last_choice)
    articles_prob[last_choice] = articles_prob[last_choice] + reward



def epsilon_greedy(choices):
    random = np.random.uniform()
    #If random < epsilon, we simply pick a random article
    choice = None
    if random < epsilon:
        article_index = int(np.random.uniform(0,len(choices)))
        choice = choices[article_index]
    else:
        max = -1
        for article in choices:
            if articles_prob[article] > max:
                choice = article
                max = articles.get(article)
        if max == 0:
            choice = articles.get(int(np.random.uniform(0,len(choices))))

    global last_choice
    last_choice = choice
    return choice


def recommend(time, user_features, choices):
    #print("choices: ", choices)
    choice = epsilon_greedy(choices)
    return choice
