import csv
import pandas as pd
import numpy as np
import random
from pprint import pprint as pp
import math


MAX_DEPTH = 3


class Leaf:
    def __init__(self, examples):
        self.predictions = count_class(examples)


class FeatureNode:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        return val >= self.value  # mb '>=' for later real values

    def __repr__(self):
        return "Q: Is %s >= %s?" % (headers[self.column], str(self.value))


df = pd.read_csv('little_data.tsv', delimiter='\t', encoding='utf-8')
headers = df.dtypes.index.values  # list of strings - names of features (all)
data_feature_names = headers[:-1]
data = df[data_feature_names].values
target_name = headers[-1]
target = df[target_name].values
# print("data:")
# print(data)
# print('target:')
# print(target)


def generate_sets():
    train_size = int(len(data) * 0.7)
    train_ind = sorted(random.sample(range(len(data)), train_size))
    test_ind = [i for i in range(len(data)) if i not in train_ind]

    train_data = [data[i] for i in train_ind]
    train_target = [target[i] for i in train_ind]
    train_set = [train_data, train_target]

    test_data = [data[i] for i in test_ind]
    test_target = [target[i] for i in test_ind]
    test_set = [test_data, test_target]

    # for i in range(len(train_set[0])):
    #    print(str(train_set[0][i]) + " => " + str(train_set[1][i]))
    return train_set, test_set


def count_class(examples):
    # pp(examples)
    counts = {}
    targets = examples[1]
    for t in targets:
        if t not in counts:
            counts[t] = 1
        else:
            counts[t] += 1
    # print(counts)
    return counts


def divide_data(examples, question):
    true_branch, false_branch = [], []
    values = examples[0]
    targets = examples[1]
    true_ind, false_ind = [], []
    for i in range(len(examples[0])):
        ex = values[i]
        if question.match(ex):
            true_branch.append(ex)
            true_ind.append(i)
        else:
            false_branch.append(ex)
            false_ind.append(i)
    true_branch = [true_branch, [targets[i] for i in true_ind]]
    false_branch = [false_branch, [targets[i] for i in false_ind]]
    return true_branch, false_branch


def entropy(examples):
    in_class, not_in_class = 0, 0
    # print('\ndata')
    # print(data)
    # print(target)
    # print('ex in entr')
    # pp(examples)
    targets = examples[1]
    # print('tar')
    # print(targets)
    whole_size = len(examples[0])
    # print('whole size:' + str(whole_size))
    for t in targets:
        if t == 1:
            in_class += 1
        else:
            not_in_class += 1
    # print('in class: ' + str(in_class) + ", not in class: " + str(not_in_class))
    if in_class == 0 or not_in_class == 0:
        return 0
    p_in = in_class / whole_size
    p_not_in = not_in_class / whole_size

    ent = -p_in * math.log2(p_in) - p_not_in * math.log2(p_not_in)
    return ent


def info_gain(true_branch, false_branch, whole_entropy):
    sum_ent = 0
    whole_size = len(true_branch[0]) + len(false_branch[0])
    branches = [true_branch, false_branch]
    # print('\nbranches:')
    # pp(branches)
    # print('whole_entropy: ' + str(whole_entropy))
    for b in branches:
        this_data = b[0]
        # pp(data)
        cur_size = len(this_data)
        # print('cur size: ' + str(cur_size))
        # print(b)
        e = entropy(b)
        # print('cur entropy: ' + str(e))
        sum_ent += cur_size * e / whole_size
    info = whole_entropy - sum_ent
    # print('total info: ' + str(info))
    return info


def split_info(true_branch, false_branch):
    len_t = float(len(true_branch[0]))
    len_f = float(len(false_branch[0]))
    len_whole = float(len_t + len_f)
    sum_info = len_t / len_whole * math.log2(len_t / len_whole) + len_f / len_whole * math.log2(len_f / len_whole)
    return -sum_info


def gain_ratio(true_branch, false_branch, whole_entropy):
    gain = info_gain(true_branch, false_branch, whole_entropy)
    split = split_info(true_branch, false_branch)
    return gain / split


def best_question(examples):
    n_features = len(data_feature_names)
    best_gain_ratio = 0
    best_q = None
    cur_entropy = entropy(examples)
    # print("examples: ")
    # pp(examples)
    for col in range(1, n_features):
        distinct_val = sorted(set(ex[col] for ex in examples[0]))
        # print("dist vals: ")
        # print(distinct_val)
        for val in distinct_val:
            q = Question(col, val)
            true_branch, false_branch = divide_data(examples, q)
            if len(true_branch[0]) == 0 or len(false_branch[0]) == 0:
                continue
            cur_gain_ratio = gain_ratio(true_branch, false_branch, cur_entropy)
            if cur_gain_ratio > best_gain_ratio:
                best_gain_ratio = cur_gain_ratio
                best_q = q
    return best_gain_ratio, best_q


def build_tree(examples, cur_depth):
    # print('\n')
    # pp(examples)
    # print('===============')
    if cur_depth == MAX_DEPTH:
        print("max depth reached!")
        return Leaf(examples)
    gain, question = best_question(examples)
    if gain == 0:
        return Leaf(examples)

    true_branch, false_branch = divide_data(examples, question)
    # pp(true_branch)
    # pp(false_branch)
    true_branch = build_tree(true_branch, cur_depth + 1)
    false_branch = build_tree(false_branch, cur_depth + 1)
    return FeatureNode(question, true_branch, false_branch)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


def print_tree(node, delim=''):
    if isinstance(node, Leaf):
        print(delim + "Predict", print_leaf(node.predictions))
        return

    print(delim + str(node.question))
    print(delim + '--> True:')
    print_tree(node.true_branch, delim + "  ")
    print(delim + '--> False:')
    print_tree(node.false_branch, delim + "  ")


def classify(dataset, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(dataset):
        return classify(dataset, node.true_branch)
    else:
        return classify(dataset, node.false_branch)


def question_number(question):
    num = np.where(data_feature_names == question)
    return num[0][0]


def main():
    train_set, test_set = generate_sets()
    # num = question_number('has_sport')
    # info_gain(train_set, num)
    # best_question(train_set)
    root = build_tree(train_set, 0)
    print_tree(root)
    test_data = test_set[0]
    test_target = test_set[1]
    print("test set: ")
    pp(test_set)
    for i in range(len(test_set[0])):
        c = classify(test_data[i], root)
        print("Real: %s. Predicted: %s" % (test_target[i], print_leaf(c)))
    # pp(test_set[0][0])


main()
