import math

def softmax(scores: list[float]) -> list[float]:

    exp_scores = [math.exp(x-max(scores)) for x in scores]
    total = sum(exp_scores)
    return [round(x / total, 4) for x in exp_scores]