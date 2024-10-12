from numpy import log2, array


def compute_entropy(y):
    labels = {}
    for example in y:
        if example not in labels.keys():
            labels[example] = 0
        labels[example] += 1
    probs = [int(labels[key]) / len(y) for key in labels.keys()]
    return - (array(probs) * log2(probs)).sum()


def compute_gini_index(y):
    labels = {}
    for example in y:
        if example not in labels.keys():
            labels[example] = 0
        labels[example] += 1
    probs = [int(labels[key]) / len(y) for key in labels.keys()]
    return (1 - (array(probs)**2).sum())


def compute_majority_error(y):
    labels = {}
    for example in y:
        if example not in labels.keys():
            labels[example] = 0
        labels[example] += 1
    if len(list(labels.values())) == 0:
        return 0
    label_counts = array(list(labels.values()))
    n_majority = label_counts.max()
    return (len(y) - n_majority) / len(y)