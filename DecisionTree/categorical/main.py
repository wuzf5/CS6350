import csv
import argparse
from numpy import array
import pandas as pd

from purity_compute import compute_entropy, compute_gini_index, compute_majority_error 


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--purity_measure", type=str, default="entropy", help='[entropy, gini, majority]')
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--test_set", type=str, default='test', help='[test, train]')
    parser.add_argument("--attribute_list", default=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
    parser.add_argument("--value_list", default={'buying': ['vhigh', 'high', 'med', 'low'], 'maint': ['vhigh', 'high', 'med', 'low'], 
                                                 'doors': ['2', '3', '4', '5more'], 'persons': ['2', '4', 'more'], 
                                                 'lug_boot': ['small', 'med', 'big'], 'safety': ['low', 'med', 'high']})
    return parser.parse_args()


def select_next_attribute(dataset, purity_measure):
    expected_ents = []
    for attr in dataset.columns[:-1]:
        feature_value_list = dataset[attr]
        probs, ents = [], []
        for value in set(feature_value_list.values.tolist()):
            subset = dataset[dataset[attr] == value].drop(attr, axis=1)
            probs.append(len(subset) / len(dataset))
            ents.append(purity_measure(subset[subset.columns[-1]].values.tolist()))
        expected_ent = (array(probs) * array(ents)).sum()
        expected_ents.append(expected_ent)
    
    return array(expected_ents).argmin() # select the index with the minimal expected entropy (i.e., maximal Gain)


def create_tree(dataset, attributes, value_list, purity_measure, max_depth, depth=0):
    label_list = dataset['label']
    # stopping criterion: all labels in the subset are the same, or the tree has reached its pre-defined maximum depth
    if len(set(label_list)) == 1:
        return label_list.iloc[0]
    if max_depth is not None and depth == max_depth:
        most_common_label = max(set(label_list), key=label_list.values.tolist().count)
        return most_common_label
    
    next_attr_idx = select_next_attribute(dataset, purity_measure)
    new_branch = attributes[next_attr_idx]
    # create a new branch on the tree
    tree = {new_branch:{}}
    del attributes[next_attr_idx]
    feature_value_list = dataset[new_branch] # list of values of the corres feature (in the subset)
    for value in set(feature_value_list.values.tolist()):
        subset = dataset[dataset[new_branch] == value].drop(new_branch, axis=1)
        tree[new_branch][value] = create_tree(subset, attributes.copy(), value_list, purity_measure, max_depth, depth+1)
        # complete the label for missing values for better generalization
        # if isinstance(tree[new_branch][value], str) and len(feature_value_list) < len(value_list[new_branch]):
        #     most_common_label = max(set(subset['label']), key=subset['label'].values.tolist().count)
        #     for value in value_list[new_branch]:
        #         if value not in feature_value_list:
        #             tree[new_branch][value] = most_common_label
    return tree


def main(args):
    data = []
    with open('../../Datasets/car/train.csv', mode ='r') as file:
        file = csv.reader(file)
        for lines in file:
            data.append(lines) # (1000, 7)
    columns_names = args.attribute_list.copy()
    columns_names.append('label')
    data = pd.DataFrame(data, columns=columns_names)

    purity_measurements = {'entropy': compute_entropy, 'gini': compute_gini_index, 'majority': compute_majority_error}
    tree = create_tree(data, args.attribute_list.copy(), args.value_list, purity_measurements[args.purity_measure], args.max_depth)
    # print(tree)
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
        

def test(tree, attribute_list, testset):
    data_count = 0
    correct_pred_count = 0
    if testset == 'train':
        file_name = '../../Datasets/car/train.csv'
    else:
        file_name = '../../Datasets/car/test.csv'
    with open(file_name, mode ='r') as file:
        file = csv.reader(file)
        for lines in file:
            prediction = predict(tree, attribute_list, lines[:-1])
            data_count += 1
            if prediction == lines[-1]:
                correct_pred_count += 1
    return round(1 - correct_pred_count / data_count, 3)


if __name__ == '__main__':
    args = get_arguments()
    tree = main(args)
    test_error = test(tree, args.attribute_list, args.test_set)
    print('prediction error: {}'.format(test_error))
