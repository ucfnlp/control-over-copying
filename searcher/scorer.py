import math


def length_norm(score):
    length_tgt = len(score)
    return sum(score) / length_tgt


def word_reward(score, reward):
    length_tgt = len(score)
    return sum(score) - reward * length_tgt


def bounded_word_reward(score, reward, bound):
    """
        bound = L_predict
        L_predict could be:
        1) length_src * alpha
        2) average length_tgt * beta
        3) model predicted length * gamma
    """
    length_tgt = len(score)
    bounded_length = min(length_tgt, bound)
    return sum(score) - reward * bounded_length


def bounded_adaptive_reward(score, rewards, bound):
    if len(rewards) > bound:
        rewards = rewards[:bound]
    return sum(score) - sum(rewards)


def neg_sigmoid(x):
    return 1.0 / (1 + math.exp(x))
