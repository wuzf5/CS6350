import numpy as np
import pandas as pd

from purity_compute import compute_entropy, compute_gini_index, compute_majority_error

class ID3DecisionTree:
    def __init__(self, args):
        self.max_depth = args.max_depth
        purity_measurements = {'entropy': compute_entropy, 'gini': compute_gini_index, 'majority': compute_majority_error}
        self.purity_measure = purity_measurements[args.purity_measure]
        self.attribute_list = args.attribute_list
        # self.value_list_set = args.value_list_set
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
                # self.value_list_set[col] = ['<={}'.format(median_value), '>{}'.format(median_value)]
        return medians

    def _create_tree(self, dataset, labels, depth=0):
        if len(set(labels)) == 1:
            return labels.iloc[0]
        
        if self.max_depth and depth == self.max_depth:
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
                # values = [v if not isinstance(v, dict) else self._most_common_label(v) for v in tree[feature].values()]
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
    
    def test(self, testdata):
        testdata.columns = columns
        test_feature_set = testdata.drop('label', axis=1)
        test_label_set = testdata['label']

        correct_pred_count = 0
        for i in range(len(test_feature_set)):
            test_feature = test_feature_set.iloc[i]
            prediction = self._predict(test_feature, self.tree)
            if not isinstance(prediction, str):
                print(11, prediction)
            # print(test_label_set.iloc[i])
            if prediction == test_label_set.iloc[i]:
                correct_pred_count += 1
        return round(1 - correct_pred_count / len(testdata), 3)


if __name__ == '__main__':
    import argparse

    def get_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument("--purity_measure", type=str, default="entropy", help='[entropy, gini, majority]')
        parser.add_argument("--max_depth", type=int, default=16)
        parser.add_argument("--test_set", type=str, default='test', help='[test, train]')
        parser.add_argument("--attribute_list", default=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                                                        'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'])
        return parser.parse_args()
    args = get_arguments()

    columns = args.attribute_list.copy()
    columns.append('label')
    traindata = pd.read_csv('../../Datasets/bank/train.csv', header=None)
    traindata.columns = columns
    trainset = traindata.drop('label', axis=1)
    trainlabels = traindata['label']

    tree = ID3DecisionTree(args)
    tree.train(trainset, trainlabels)
    # testing
    testdata = pd.read_csv('../../Datasets/bank/test.csv', header=None) if args.test_set == 'test' else pd.read_csv('../../Datasets/bank/train.csv')
    print('prediction error: {}'.format(tree.test(testdata)))
    # print(' {} & '.format(tree.test(testdata)))

    # train_acc, test_acc = [], []
    # for i in range(16):
    #     train_acc.append([])
    #     test_acc.append([])
    #     for j, measure in enumerate(['entropy', 'gini', 'majority']):
    #         for test in ['train', 'test']:
    #             args.max_depth = i + 1
    #             args.purity_measure = measure
    #             args.test_set = test

    #             tree = ID3DecisionTree(args)
    #             tree.train(trainset, trainlabels)

    #             # testing
    #             testdata = pd.read_csv('./bank/test.csv') if args.test_set == 'test' else pd.read_csv('./bank/train.csv')
    #             if args.test_set == 'test':
    #                 test_acc[i].append(tree.test(testdata))
    #             else:
    #                 train_acc[i].append(tree.test(testdata))
    # train_avg = np.array(train_acc).mean(axis=1)
    # np.round(train_avg, 3)
    # test_avg = np.array(test_acc).mean(axis=1)
    # np.round(test_avg, 3)
    # print(np.array(train_acc).shape)
    # print('training error: {}'.format(train_acc))
    # print('average training error: {}'.format(train_avg))
    # print('testing error: {}'.format(test_acc))
    # print('average testing error: {}'.format(test_avg))
