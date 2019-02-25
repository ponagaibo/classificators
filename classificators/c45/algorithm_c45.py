import pandas as pd
import numpy as np
import random
from pprint import pprint as pp
import math


MAX_DEPTH = 6
headers = None


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
        return val >= self.value

    def __repr__(self):
        return "Q: Is %s >= %s?" % (headers[self.column], str(self.value))


def load_tsv():
    df = pd.read_csv('little_data.tsv', delimiter='\t', encoding='utf-8')
    global headers
    headers = df.dtypes.index.values
    data_feature_names = headers[:-1]
    data = df[data_feature_names].values
    target_name = headers[-1]
    target = df[target_name].values
    return data, target


def generate_sets(data, target):
    train_size = int(len(data) * 0.7)
    train_ind = sorted(random.sample(range(len(data)), train_size))
    test_ind = [i for i in range(len(data)) if i not in train_ind]

    train_data = [data[i] for i in train_ind]
    train_target = [target[i] for i in train_ind]
    train_set = [train_data, train_target]

    test_data = [data[i] for i in test_ind]
    test_target = [target[i] for i in test_ind]
    test_set = [test_data, test_target]
    return train_set, test_set


def count_class(examples):
    counts = {}
    targets = examples[1]
    for t in targets:
        if t not in counts:
            counts[t] = 1
        else:
            counts[t] += 1
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
    targets = examples[1]
    whole_size = len(examples[0])
    # only two classes - if 1 => in class, not in class otherwise
    for t in targets:
        if t == 1:
            in_class += 1
        else:
            not_in_class += 1
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
    for b in branches:
        this_data = b[0]
        cur_size = len(this_data)
        e = entropy(b)
        sum_ent += cur_size * e / whole_size
    info = whole_entropy - sum_ent
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


def best_question(examples, rf=False):
    data_feature_names = headers[1:-1]
    n_features = len(data_feature_names)

    best_gain_ratio = 0
    best_q = None
    cur_entropy = entropy(examples)  # ???

    if rf:
        subfeatures_len = round(math.sqrt(n_features))
        cur_features_num = sorted(random.sample([i for i in range(1, n_features + 1)], subfeatures_len))
    else:
        cur_features_num = [i for i in range(1, n_features + 1)]
    for col in cur_features_num:
        distinct_val = sorted(set(ex[col] for ex in examples[0]))
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


def build_tree(examples, cur_depth, rf=False):
    if cur_depth == MAX_DEPTH:
        print("max depth reached!")
        return Leaf(examples)
    gain, question = best_question(examples, rf)
    if gain == 0:
        return Leaf(examples)

    true_branch, false_branch = divide_data(examples, question)
    true_branch = build_tree(true_branch, cur_depth + 1, rf)
    false_branch = build_tree(false_branch, cur_depth + 1, rf)
    return FeatureNode(question, true_branch, false_branch)


def printable_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    # for kk, vv in counts.items():
    #     print(str(kk) + " : " + str(vv))
    for lbl in counts.keys():
        probs[lbl] = str(round(counts[lbl] / total * 100)) + "%"
    return probs


def main_class(counts):
    maxVal = -1
    mainClass = -1
    for k, v in counts.items():
        if v > maxVal:
            maxVal = v
            mainClass = k
    return mainClass


def print_tree(node, delim=''):
    if isinstance(node, Leaf):
        print(delim + "Predict", printable_leaf(node.predictions))
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
    data_feature_names = headers[:-1]
    num = np.where(data_feature_names == question)
    return num[0][0]


def main():
    # for every tree choose subset and build three
    # when trees are built, use test set to everyone and collect answers
    # most popular answer is a global answer
    num_of_trees = 10
    data, target = load_tsv()
    train_set, test_set = generate_sets(data, target)

    list_of_trees = []

    for _ in range(num_of_trees):
        build_rf = True
        sub_data = []
        sub_target = []
        for _ in train_set[0]:
            k = random.randrange(0, len(train_set[0]))
            sub_data.append(train_set[0][k])
            sub_target.append(train_set[1][k])
        sub_train_set = [sub_data, sub_target]

        root = build_tree(sub_train_set, 0, build_rf)
        # print_tree(root)
        list_of_trees.append(root)

    test_data = test_set[0]
    test_target = test_set[1]
    print("test set: ")
    pp(test_set)

    all_classes = []
    for i in range(len(test_set[0])):
        total_classes = {}
        tree_cnt = 1
        for tree in list_of_trees:
            c = classify(test_data[i], tree)
            # print("tree #" + str(tree_cnt))
            # print(str(c) + " => ", end="")
            # print(printable_leaf(c))
            for kk, vv in c.items():
                if kk not in total_classes:
                    total_classes[kk] = 0
                total_classes[kk] += vv
            tree_cnt += 1
        all_classes.append(total_classes)

        print("For %s real: %s. Predicted: %s"
              % (test_data[i][0], test_target[i], main_class(total_classes)))

    print()
    for i in range(len(test_set[0])):
        print("For %s real: %s. Predicted: %s"
              % (test_data[i][0], test_target[i], main_class(all_classes[i])))

main()
