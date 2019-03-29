import pandas as pd
import numpy as np
import random
from pprint import pprint as pp
import math


MAX_DEPTH = 2
headers = None


class Leaf:
    def __init__(self, examples=None, pred1=None, pred2=None, err=None):
        if examples is not None:
            self.predictions = count_class(examples)
            self.errors = 0.0
        elif pred1 is not None and pred2 is not None:
            print("pruned leaf")
            d = dict(pred1)
            for k, v in pred2.items():
                if k not in d:
                    d[k] = v
                else:
                    d[k] += v
            self.predictions = dict(d)
            self.errors = err
        else:
            print("Error!!! Wrong constructor arguments")


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
    # print("in count classes, elems:")
    # for el in examples[0]:
    #     print(el)
    # print("targets:")
    # for tar in examples[1]:
    #     print(tar)

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


def best_question(examples):
    data_feature_names = headers[1:-1]
    n_features = len(data_feature_names)

    best_gain_ratio = 0
    best_q = None
    cur_entropy = entropy(examples)  # ???
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


def build_tree(examples, cur_depth):
    if cur_depth == MAX_DEPTH:
        print("max depth reached!")
        return Leaf(examples)
    gain, question = best_question(examples)
    if gain == 0:
        return Leaf(examples)

    true_branch, false_branch = divide_data(examples, question)
    true_branch = build_tree(true_branch, cur_depth + 1)
    false_branch = build_tree(false_branch, cur_depth + 1)
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
        print(delim + "Predict " + str(node.predictions) + ", er: " + str(node.errors))
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


def count_errors(dataset, target, node):
    if isinstance(node, Leaf):
        # print("\ncur dataset:")
        # pp(dataset)
        # print("cur target: ", target)
        # print("cur leaf: ", node.predictions, sum(node.predictions.values()))
        # print("main class: ", main_class(node.predictions))
        if target != main_class(node.predictions):
            node.errors += 1
        return node.predictions

    if node.question.match(dataset):
        return count_errors(dataset, target, node.true_branch)
    else:
        return count_errors(dataset, target, node.false_branch)

def question_number(question):
    data_feature_names = headers[:-1]
    num = np.where(data_feature_names == question)
    return num[0][0]


def count_and_prune(tree, train_set):
    for i in range(len(train_set[0])):
        c = count_errors(train_set[0][i], train_set[1][i], tree)

    print("\nafter count errors:")
    print_tree(tree)
    print("\nprunning...")
    prune(tree)
    print("\nafter pruning: ")
    print_tree(tree)


def prune(tree):
    z_a = 1.64

    if not isinstance(tree.true_branch, Leaf):
        tree.true_branch = prune(tree.true_branch)
    if not isinstance(tree.false_branch, Leaf):
        tree.false_branch = prune(tree.false_branch)

    if isinstance(tree.true_branch, Leaf) and isinstance(tree.false_branch, Leaf):
        # print("cur node with q: " + str(tree.question))
        true_size = sum(tree.true_branch.predictions.values())
        t_leaf_er = tree.true_branch.errors / true_size
        true_err = t_leaf_er + z_a * math.sqrt((t_leaf_er * (1 - t_leaf_er)) / true_size)

        false_size = sum(tree.false_branch.predictions.values())
        f_leaf_er = tree.false_branch.errors / false_size
        false_err = f_leaf_er + z_a * math.sqrt((f_leaf_er * (1 - f_leaf_er)) / false_size)

        # print("true leaf: {}, er: {}, s: {}, pes: {}".format(tree.true_branch.predictions, tree.true_branch.errors,
        #                                                      true_size, true_err))
        # print("false leaf: {}, er: {}, s: {}, pes: {}".format(tree.false_branch.predictions, tree.false_branch.errors,
        #                                                       false_size, false_err))

        total_size = true_size + false_size
        pes_err = (true_size * true_err + false_size * false_err) / total_size
        cur_err = (tree.true_branch.errors + tree.false_branch.errors) / total_size
        # print("total size: {}, pes_err: {}, cur_err = {}".format(total_size, pes_err, cur_err))

        leafs_err = t_leaf_er + f_leaf_er

        if cur_err > pes_err:
            print("do pruning!!")
            return Leaf(None, tree.true_branch.predictions, tree.false_branch.predictions, leafs_err)
        else:
            # print("ordinary tree")
            return tree
    return tree


def main():
    data, target = load_tsv()
    train_set, test_set = generate_sets(data, target)

    root = build_tree(train_set, 0)
    # print_tree(root)

    train_data = train_set[0]
    train_target = train_set[1]

    test_data = test_set[0]
    test_target = test_set[1]
    # print("test set: ")
    # pp(test_set)
    # print("train set: ")
    # pp(train_set)

    all_classes = []
    for i in range(len(test_set[0])):
        c = classify(test_data[i], root)
        all_classes.append(c)

    # for i in range(len(train_set[0])):
    #     c = count_errors(train_data[i], train_target[i], root)
    #
    # print_tree(root)
    count_and_prune(root, train_set)

    for i in range(len(test_set[0])):
        print("For %s real: %s. Predicted: %s"
              % (test_data[i][0], test_target[i], main_class(all_classes[i])))

main()
