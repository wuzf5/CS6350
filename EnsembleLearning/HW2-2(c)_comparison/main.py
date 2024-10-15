import numpy as np
import pandas as pd

from purity_compute import compute_entropy, compute_gini_index, compute_majority_error


class ID3DecisionTree:
    def __init__(self, args):
        self.max_depth = args.max_depth
        purity_measurements = {'entropy': compute_entropy, 'gini': compute_gini_index, 'majority': compute_majority_error}
        self.purity_measure = purity_measurements[args.purity_measure]
        self.attribute_list = args.attribute_list
        self.value_list_set = args.value_list_set
        self.min_split_size = args.min_split_size
        self.tree = None

    def train(self, dataset, labels):
        self.features = list(dataset.columns)
        self.feature_types = self._get_feature_types(dataset)
        self.medians = self._get_medians_and_set_up_numeric_value_in_value_lst(dataset)
        self.tree = self._create_tree(dataset, labels)

    def _get_feature_types(self, dataset):
        feature_types = {}
        for col in dataset.columns:
            if dataset[col].dtype == 'object':
                feature_types[col] = 'categorical'
            else:
                feature_types[col] = 'numeric'
        return feature_types

    def _get_medians_and_set_up_numeric_value_in_value_lst(self, dataset):
        medians = {}
        for col in dataset.columns:
            if self.feature_types[col] == 'numeric':
                median_value = dataset[col].median()
                medians[col] = median_value
                self.value_list_set[col] = ['<={}'.format(median_value), '>{}'.format(median_value)]
        return medians

    def _create_tree(self, dataset, labels, depth=0):
        if len(set(labels)) == 1:
            return labels.iloc[0]

        if (self.max_depth and depth == self.max_depth):# or dataset.shape[0] <= self.min_split_size:
            most_common_label = labels.reset_index(drop=True).mode().iloc[0]
            return most_common_label

        best_feature = self._select_next_attribute(dataset, labels)
        tree = {best_feature: {}}

        if self.feature_types[best_feature] == 'categorical':
            for value in set(dataset[best_feature].values.tolist()):
                subset_features = dataset[dataset[best_feature] == value].drop(best_feature, axis=1)
                subset_labels = labels[dataset[best_feature] == value]

                if len(subset_features) == 0:
                    tree[best_feature][value] = labels.reset_index(drop=True).mode().iloc[0]
                else:
                    tree[best_feature][value] = self._create_tree(subset_features, subset_labels, depth+1)
            tree[best_feature]['not_included_in_training_data'] = labels.reset_index(drop=True).mode().iloc[0]
        else:  # numeric
            pivot_value = self.medians[best_feature]
            for condition in ['<=', '>']:
                value = '{}{}'.format(condition, pivot_value)
                if condition == '<=':
                    subset_features = dataset[dataset[best_feature] <= pivot_value].drop(best_feature, axis=1)
                    subset_labels = labels[dataset[best_feature] <= pivot_value]
                else:
                    subset_features = dataset[dataset[best_feature] > pivot_value].drop(best_feature, axis=1)
                    subset_labels = labels[dataset[best_feature] > pivot_value]
                if subset_features.empty:
                    tree[best_feature][value] = labels.reset_index(drop=True).mode().iloc[0]
                else:
                    tree[best_feature][value] = self._create_tree(subset_features, subset_labels, depth + 1)
        return tree

    def _select_next_attribute(self, dataset, labels):
        expected_ents = []
        for feature in dataset.columns:
            feature_value_list = dataset[feature]
            if self.feature_types[feature] == 'categorical':
                probs, ents = [], []
                for value in set(feature_value_list.values.tolist()):
                    probs.append(len(labels[feature_value_list == value]) / len(labels))
                    ents.append(self.purity_measure(labels[feature_value_list == value]))
                expected_ent = (np.array(probs) * np.array(ents)).sum()
            else:  # numeric
                median = self.medians[feature]
                left_subset, right_subset = labels[feature_value_list <= median], labels[feature_value_list > median]
                left_ent, right_ent = self.purity_measure(left_subset), self.purity_measure(right_subset)
                left_prob, right_prob = len(left_subset) / len(labels), len(right_subset) / len(labels)
                expected_ent = left_ent * left_prob + right_ent * right_prob
            expected_ents.append(expected_ent)
        idx = np.array(expected_ents).argmin()
        return dataset.columns[idx] # select the feature that yields the minimal expected entropy (i.e., maximal Gain)

    def _predict(self, row, tree):
        if not isinstance(tree, dict):
            return tree
        feature = list(tree.keys())[0]
        if self.feature_types[feature] == 'categorical':
            if row[feature] in tree[feature]:
                return self._predict(row, tree[feature][row[feature]])
            else: # return most common feature
                most_common_label = max(tree[feature].values(), key=lambda x: list(tree[feature].values()).count(x))
                if isinstance(most_common_label, dict): # an attribute value that hasn't been encountered in training
                    return tree[feature]['not_included_in_training_data']
                return most_common_label
        else:  # numeric
            median = self.medians[feature]
            if row[feature] <= median:
                return self._predict(row, tree[feature]['<={}'.format(median)])
            else:
                return self._predict(row, tree[feature]['>{}'.format(median)])

    def predict_batch(self, feature_batch):
        # print(feature_batch.shape) # (4999, 16)
        predictions = []
        for i in range(len(feature_batch)):
            test_feature = feature_batch.iloc[i]
            prediction = self._predict(test_feature, self.tree)
            predictions.append(prediction)
        return np.stack(predictions)


class BaggedTrees:
    def __init__(self, args):
        self.n_iters = int(args.n_iters)
        self.args = args
        self.trees = []

    def _sample_data_with_replacement(self, dataset):
        sampled_dataset = dataset.sample(dataset.shape[0], axis=0, replace=True).reset_index(drop=True)
        features = sampled_dataset.drop('label', axis=1)
        labels = sampled_dataset['label']
        return features, labels

    def train(self, dataset):
        for t in range(self.n_iters):
            features, labels = self._sample_data_with_replacement(dataset)
            tree = ID3DecisionTree(self.args)
            tree.train(features, labels)
            self.trees.append(tree)
    
    def predict(self, features):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict_batch(features))
        final_predictions = pd.DataFrame(predictions).mode().iloc[0]
        return final_predictions


if __name__ == '__main__':
    import argparse
    from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--purity_measure", type=str, default="entropy", help='[entropy, gini, majority]')
    parser.add_argument("--max_depth", type=int, default=16)
    parser.add_argument("--test_set", type=str, default='test', help='[test, train]')
    parser.add_argument("--attribute_list", default=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                                                    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'])
    parser.add_argument("--n_iters", type=int, default=500)
    parser.add_argument("--n_bags", type=int, default=100)
    parser.add_argument("--min_split_size", type=int, default=50)
    parser.add_argument("--value_list_set", default={'age': None, \
                                                     'job': ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"], \
                                                     'marital': ["married","divorced","single"], \
                                                     'education': ["unknown","secondary","primary","tertiary"], \
                                                     'default': ["yes","no"], \
                                                     'balance': None, \
                                                     'housing': ["yes","no"], \
                                                     'loan': ["yes","no"], \
                                                     'contact': ["unknown","telephone","cellular"], \
                                                     'day': None, 
                                                     'month': ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], \
                                                     'duration': None, \
                                                     'campaign': None, \
                                                     'pdays': None, \
                                                     'previous': None, 
                                                     'poutcome': ["unknown","other","failure","success"]})

    args = parser.parse_args()

    columns = args.attribute_list.copy()
    columns.append('label')
    traindata = pd.read_csv('../../Datasets/bank/train.csv', header=None)
    

    # testing data processing
    testdata = pd.read_csv('../../Datasets/bank/test.csv', header=None) # if args.test_set == 'test' else pd.read_csv('../bank/train.csv')
    testdata.columns = columns
    test_features = testdata.drop('label', axis=1)
    test_labels = testdata['label']
    test_labels[test_labels == 'yes'] = 1.
    test_labels[test_labels == 'no'] = 0.

    bags = []
    single_trees = []
    training_errors = []
    testing_errors = []
    for i in range(1, args.n_bags+1):
        traindata = traindata.sample(1000, axis=0, replace=False).reset_index(drop=True)
        traindata.columns = columns
        train_features = traindata.drop('label', axis=1)
        train_labels = traindata['label']
        train_labels[train_labels == 'yes'] = 1.
        train_labels[train_labels == 'no'] = 0.
        bagged_trees = BaggedTrees(args)
        bagged_trees.train(traindata)
        bags.append(bagged_trees)
        single_trees.append(bagged_trees.trees[0])

    single_tree_predictions = []
    for single_tree in single_trees:
        single_tree_predictions.append(single_tree.predict_batch(test_features))
    single_tree_predictions = np.stack(single_tree_predictions) # (n_bags, 4999)
    avg_single_tree_pred = single_tree_predictions.mean(axis=0, keepdims=True) # (1, 4999)
    single_tree_bias = (np.squeeze(avg_single_tree_pred) - test_labels) ** 2
    avg_single_tree_bias = single_tree_bias.mean()
    print('(SINGLE TREE) average bias w.r.t. all test samples: ', avg_single_tree_bias)
    single_tree_sample_var = (1 / (args.n_bags-1)) * ((single_tree_predictions - avg_single_tree_pred) ** 2).sum(axis=0)
    avg_single_tree_sample_var = single_tree_sample_var.mean()
    print('(SINGLE TREE) average var w.r.t. all test samples: ', avg_single_tree_sample_var)
    single_general_squared_error = avg_single_tree_bias + avg_single_tree_sample_var
    print('(SINGLE TREE) general squared error w.r.t. test examples: ', single_general_squared_error)

    bagged_tree_predictions = []
    for bag in bags:
        bagged_tree_predictions.append(bag.predict(test_features).to_numpy()) # (4999,)
    bagged_tree_predictions = np.stack(bagged_tree_predictions)
    avg_bagged_tree_pred = bagged_tree_predictions.mean(axis=0, keepdims=True) # (1, 4999)
    bagged_tree_bias = (np.squeeze(avg_bagged_tree_pred) - test_labels) ** 2
    avg_bagged_tree_bias = bagged_tree_bias.mean()
    print('(BAGGED TREE) average bias w.r.t. all test samples: ', avg_bagged_tree_bias)
    bagged_tree_sample_var = (1 / (args.n_bags-1)) * ((bagged_tree_predictions - avg_bagged_tree_pred) ** 2).sum(axis=0)
    avg_bagged_tree_sample_var = bagged_tree_sample_var.mean()
    print('(BAGGED TREE) average var w.r.t. all test samples: ', avg_bagged_tree_sample_var)
    bagged_general_squared_error = avg_bagged_tree_bias + avg_bagged_tree_sample_var
    print('(BAGGED TREE) general squared error w.r.t. test examples: ', bagged_general_squared_error)
