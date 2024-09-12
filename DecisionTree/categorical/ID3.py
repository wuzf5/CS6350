import csv
import argparse
from numpy import array
import pandas as pd

from purity_compute import compute_entropy, compute_gini_index, compute_majority_error 


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--purity_measure", type=str, default="entropy", help='[entropy, gini, majority]')
    parser.add_argument("--max_depth", type=int, default=66)
    parser.add_argument("--attribute_list", default=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
    parser.add_argument("--value_list", default={'buying': ['vhigh', 'high', 'med', 'low'], 'maint': ['vhigh', 'high', 'med', 'low'], 
                                                 'doors': ['2', '3', '4', '5more'], 'persons': ['2', '4', 'more'], 
                                                 'lug_boot': ['small', 'med', 'big'], 'safety': ['low', 'med', 'high']})
    return parser.parse_args()


def get_subset(dataset, column, value):
    panda_dataset = pd.DataFrame(data=dataset)
    subset = panda_dataset[panda_dataset[column] == value]
    del subset[column]
    return subset.values.tolist()


def select_next_attribute(dataset, purity_measure):
    expected_ents = []
    for attr in range(len(dataset[0])-1):
        feature_value_list = pd.DataFrame(dataset)[attr].values.tolist()
        probs, ents = [], []
        for value in set(feature_value_list):
            subset = get_subset(dataset, attr, value)
            probs.append(len(subset) / len(dataset))
            ents.append(purity_measure(subset))
        expected_ent = (array(probs) * array(ents)).sum()
        expected_ents.append(expected_ent)
    
    return array(expected_ents).argmin() # select the index with the minimal expected entropy (i.e., maximal Gain)


def create_tree(dataset, attributes, value_list, purity_measure, max_depth, depth=0):
    label_list = [example[-1] for example in dataset]
    # stopping criterion: all labels in the subset are the same, or the tree has reached its pre-defined maximum depth
    if len(set(label_list)) == 1 or (max_depth is not None and depth == max_depth):
        return label_list[0]
    
    next_attr_idx = select_next_attribute(dataset, purity_measure)
    new_branch = attributes[next_attr_idx]
    # create a new branch on the tree
    tree = {new_branch:{}}
    depth += 1
    del attributes[next_attr_idx]
    feature_value_list = pd.DataFrame(dataset)[next_attr_idx].values.tolist() # list of values of the corres feature (in the subset)
    for value in set(feature_value_list):
        subset = get_subset(dataset, next_attr_idx, value)
        tree[new_branch][value] = create_tree(subset, attributes.copy(), value_list, purity_measure, max_depth, depth)
        # complete the label for missing values for better generalization
        if isinstance(tree[new_branch][value], str) and len(feature_value_list) < len(value_list[new_branch]):
            subset_label_list = [example[-1] for example in subset]
            most_common_label = max(set(subset_label_list), key=subset_label_list.count)
            for value in value_list[new_branch]:
                if value not in feature_value_list:
                    tree[new_branch][value] = most_common_label
    return tree


def main(args):
    data = []
    with open('./car/train.csv', mode ='r') as file:
        file = csv.reader(file)
        for lines in file:
            data.append(lines) # (1000, 7)

    purity_measurements = {'info': compute_entropy, 'gini': compute_gini_index, 'majority': compute_majority_error}
    tree = create_tree(data, args.attribute_list.copy(), args.value_list, purity_measurements[args.purity_measure], args.max_depth)
    print(tree)
    return tree


def predict(tree, attributes, test_feature):
    init_attr = list(tree.keys())[0]
    subset = tree[init_attr]
    idx = attributes.index(init_attr)
    for key in subset.keys():
        if test_feature[idx] == key:
            if isinstance(subset[key], dict):
                label = predict(subset[key], attributes, test_feature)
            else:
                label = subset[key]
            return label
        

def test(tree, attribute_list):
    data_count = 0
    correct_pred_count = 0
    with open('./car/test.csv', mode ='r') as file:
        file = csv.reader(file)
        for lines in file:
            prediction = predict(tree, attribute_list, lines[:-1])
            data_count += 1
            if prediction == lines[-1]:
                correct_pred_count += 1
    return correct_pred_count / data_count


if __name__ == '__main__':
    args = get_arguments()
    tree = main(args)
    test_accuracy = test(tree, args.attribute_list)
    print(test_accuracy)
