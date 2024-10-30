import numpy as np
import pandas as pd
from math import log


class DecisionStump:
    def __init__(self, args):
        self.max_depth = 1
        self.attribute_list = args.attribute_list
        self.tree = None
        self.alpha_t = None

    def train(self, features, labels, weights):
        self.features = list(features.columns)
        self.feature_types = self._get_feature_types(features)
        self.medians = self._get_medians(features)
        self.tree = self._create_tree(features, labels, weights)
        print(self.tree)

    def _compute_weighted_entropy(self, y, weights):
        labels = {}
        for i, example in enumerate(y):
            if example not in labels.keys():
                labels[example] = 0
            labels[example] += weights[i]
        total_subset_weight = weights.sum()
        probs = [labels[key] / total_subset_weight for key in labels.keys()]
        return -(np.array(probs) * np.log2(probs)).sum()

    def _select_next_attribute(self, feature_set, labels, weights):
        expected_ents = []
        for feature in feature_set.columns:
            feature_value_list = feature_set[feature]
            if self.feature_types[feature] == 'categorical':
                if len(feature_value_list.values.tolist()) == 1:
                    return 0
                probs, ents = [], []
                for value in set(feature_value_list.values.tolist()):
                    mask = feature_value_list == value
                    probs.append(weights[mask].sum() / weights.sum())
                    ents.append(self._compute_weighted_entropy(labels[mask], weights[mask]))
                expected_ent = (np.array(probs) * np.array(ents)).sum()
            else: # numeric
                median = self.medians[feature]
                left_mask, right_mask = feature_value_list <= median, feature_value_list > median
                left_subset, right_subset = labels[left_mask], labels[right_mask]
                left_ent, right_ent = self._compute_weighted_entropy(left_subset, weights[left_mask]), \
                                      self._compute_weighted_entropy(right_subset, weights[right_mask])
                left_prob, right_prob = weights[left_mask].sum() / weights.sum(), \
                                        weights[right_mask].sum() / weights.sum()
                expected_ent = left_ent * left_prob + right_ent * right_prob
            expected_ents.append(expected_ent)

        return feature_set.columns[np.array(expected_ents).argmin()] # select the feature that yields the minimal expected entropy (i.e., maximal Gain)

    def _get_feature_types(self, dataset):
        feature_types = {}
        for col in dataset.columns:
            if dataset[col].dtype == 'object':
                feature_types[col] = 'categorical'
            else:
                feature_types[col] = 'numeric'
        return feature_types

    def _get_medians(self, dataset):
        medians = {}
        for col in dataset.columns:
            if self.feature_types[col] == 'numeric':
                medians[col] = dataset[col].median()
        return medians

    def _create_tree(self, dataset, labels, weights, depth=0):
        if len(set(labels)) == 1:
            return labels.iloc[0]
        
        if self.max_depth and depth == self.max_depth:
            most_common_label = labels.reset_index(drop=True).mode().iloc[0]
            return most_common_label

        best_feature = self._select_next_attribute(dataset, labels, weights)
        print(best_feature)
        tree = {best_feature: {}}

        if self.feature_types[best_feature] == 'categorical':
            for value in set(dataset[best_feature].values.tolist()):
                subset_features = dataset[dataset[best_feature] == value].drop(best_feature, axis=1)
                subset_labels = labels[dataset[best_feature] == value]

                if len(subset_features) == 0:
                    tree[best_feature][value] = labels.reset_index(drop=True).mode().iloc[0]
                else:
                    tree[best_feature][value] = self._create_tree(subset_features, subset_labels, weights[dataset[best_feature] == value], depth+1)
            tree[best_feature]['not_included_in_training_data'] = labels.reset_index(drop=True).mode().iloc[0]
        else:  # numeric
            pivot_value = self.medians[best_feature]
            which_weights = {}
            for condition in ['<=', '>']:
                if condition == '<=':
                    subset_features = dataset[dataset[best_feature] <= pivot_value].drop(best_feature, axis=1)
                    subset_labels = labels[dataset[best_feature] <= pivot_value]
                    which_weights[condition] = weights[dataset[best_feature] <= pivot_value]
                else:
                    subset_features = dataset[dataset[best_feature] > pivot_value].drop(best_feature, axis=1)
                    subset_labels = labels[dataset[best_feature] > pivot_value]
                    which_weights[condition] = weights[dataset[best_feature] > pivot_value]
                if subset_features.empty:
                    tree[best_feature]['{}{}'.format(condition, pivot_value)] = labels.reset_index(drop=True).mode().iloc[0]
                else:
                    tree[best_feature]['{}{}'.format(condition, pivot_value)] = self._create_tree(subset_features, subset_labels, which_weights[condition], depth + 1)

        return tree

    def test(self, testdata):
        testdata.columns = columns
        test_feature_set = testdata.drop('label', axis=1)
        test_label_set = testdata['label']

        correct_pred_count = 0
        for i in range(len(test_feature_set)):
            test_feature = test_feature_set.iloc[i]
            prediction = self._predict(test_feature, self.tree)
            if prediction == test_label_set.iloc[i]:
                correct_pred_count += 1
        return round(1 - correct_pred_count / len(testdata), 3)

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
        else: # numeric
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


class AdaBoost:
    def __init__(self, args):
        self.n_iters = int(args.n_iters)
        self.args = args
    
    def _init_weights(self, data_size):
        self.importance_weights = np.ones(data_size) / data_size

    def _update_weights(self, predictions, labels):
        labels = labels.to_numpy().squeeze().astype(float)
        # print((predictions * labels).sum(), (predictions == labels).sum() - (predictions != labels).sum())
        epsilon_t = (1/2) - (1/2) * (self.importance_weights * predictions * labels).sum()
        # print(111, (predictions != labels).sum(), self.importance_weights[:(predictions != labels).sum()].sum())
        # print(self.importance_weights[predictions != labels].sum(),  epsilon_t)
        # epsilon_t = self.importance_weights[predictions != labels].sum()
        # epsilon_t = (self.importance_weights * (predictions != labels)).sum()
        # print(epsilon_t)
        # assert epsilon_t < 0.5
        alpha_t = (1/2) * log((1 - epsilon_t) / epsilon_t + 1e-10)
        # alpha_t += np.random.uniform(low=1e-8, high=1e-6)
        print(epsilon_t, alpha_t)
        self.importance_weights *= np.exp(-alpha_t * labels * predictions)
        self.importance_weights /= np.sum(self.importance_weights)
        return alpha_t

    def train(self, features, labels):
        self._init_weights(labels.shape[0]) # (4999,)
        self.stumps = []
        for t in range(self.n_iters):
            # self.old_weights = self.importance_weights.copy()
            stump = DecisionStump(self.args)
            # print('weights: {}'.format(self.importance_weights))
            stump.train(features, labels, self.importance_weights)
            predictions = stump.predict_batch(features)
            # predictions = features.apply(lambda x: stump._predict(x, stump), axis=1)
            # print((predictions == 1.).sum())
            stump.alpha_t = self._update_weights(predictions, labels)
            # print((self.old_weights != self.importance_weights).sum())
            self.stumps.append(stump)
    
    def predict(self, features):
        weighted_predictions_from_ensemble = np.array([stump.predict_batch(features) * stump.alpha_t for stump in self.stumps]).sum(axis=0) # (n_stumps, n_samples)
        return np.sign(weighted_predictions_from_ensemble)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--purity_measure", type=str, default="entropy", help='[entropy, gini, majority]')
    # parser.add_argument("--max_depth", type=int, default=16)
    parser.add_argument("--test_set", type=str, default='test', help='[test, train]')
    parser.add_argument("--attribute_list", default=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                                                    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'])
    parser.add_argument("--n_iters", type=int, default=10)
    args = parser.parse_args()

    columns = args.attribute_list.copy()
    columns.append('label')
    traindata = pd.read_csv('../../Datasets/bank/train.csv', header=None)
    # traindata = traindata.iloc[:10] # NOTE debug
    traindata.columns = columns
    # print(traindata)
    trainset = traindata.drop('label', axis=1)
    trainlabels = traindata['label']
    trainlabels[trainlabels == 'yes'] = 1.
    trainlabels[trainlabels == 'no'] = -1.

    adaboost = AdaBoost(args)
    adaboost.train(trainset, trainlabels)
    # testing
    testdata = pd.read_csv('../../Datasets/bank/test.csv') if args.test_set == 'test' else pd.read_csv('../../Datasets/bank/train.csv')
    testdata.columns = columns
    test_features = testdata.drop('label', axis=1)
    test_labels = testdata['label']
    test_labels[test_labels == 'yes'] = 1.
    test_labels[test_labels == 'no'] = -1.
    print('prediction error: {}'.format(1 - (adaboost.predict(test_features) == test_labels).mean()))
