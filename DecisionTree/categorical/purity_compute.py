from math import log
from numpy import log2, array


def compute_entropy(data):
    labels = {}
    for example in data:
        if example[-1] not in labels.keys():
            labels[example[-1]] = 0
        labels[example[-1]] += 1
    probs = [int(labels[y]) / len(data) for y in labels.keys()]
    return - (array(probs) * log2(probs)).sum()


def compute_gini_index(data):
    labels = {}
    for example in data:
        if example[-1] not in labels.keys():
            labels[example[-1]] = 0
        labels[example[-1]] += 1
    probs = [int(labels[y]) / len(data) for y in labels.keys()]
    return (1 - (array(probs)**2).sum())
    

def compute_majority_error(data):
    labels = {}
    for example in data:
        if example[-1] not in labels.keys():
            labels[example[-1]] = 0
        labels[example[-1]] += 1
    label_counts = array(list(labels.values()))
    n_majority = label_counts.max()
    return (len(data) - n_majority) / len(data)