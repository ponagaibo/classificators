import pandas as pd
from math import ceil
import numpy as np
import random
from pprint import pprint as pp
import math
import time
from loader import load_tsv


MAX_DEPTH = 30
NUM_OF_BUCKETS = 8
LEAF_SIZE = 99
UNIFORM = False


class Leaf:
    def __init__(self, examples=None, pred1=None, pred2=None, err=None):
        if examples is not None:
            self.predictions = count_class(examples)
            self.errors = 0.0
        elif pred1 is not None and pred2 is not None:
            # print("pruned leaf")
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
        return "Q: Is #%d >= %s?" % (self.column, str(self.value))


class Bucket:
    def __init__(self, start, end, label):
        self.start = start
        self.end = end
        self.label = label
        self.cnt = 0

    def add_to_cnt(self, num=1):
        self.cnt += num

    def __repr__(self):
        return "'%s': (%s : %s], %s pcs" % (str(self.label), str(self.start), str(self.end), str(self.cnt))


def generate_sets(data, target, split_coef, uniform=False):
    train_size = int(len(data) * split_coef)
    print("train set size: {}, test set size: {}".format(train_size, len(data) - train_size))
    train_ind = sorted(random.sample(range(len(data)), train_size))
    test_ind = [i for i in range(len(data)) if i not in train_ind]

    cnt0 = 0
    cnt1 = 0
    train_data = [data[i] for i in train_ind]
    train_target = []
    # train_target = [target[i] for i in train_ind]
    for i in train_ind:
        train_target.append(target[i])
        if target[i] == 0:
            cnt0 += 1
        else:
            cnt1 += 1

    while uniform and cnt0 != cnt1:
        k = random.randrange(0, len(target))
        if cnt0 > cnt1:
            if target[k] == 1:
                train_data.append(data[k][:])
                train_target.append(target[k])
                cnt1 += 1
        else:
            if target[k] == 0:
                train_data.append(data[k][:])
                train_target.append(target[k])
                cnt0 += 1
    print("uniform: {}, in train set: cnt0 = {}, cnt1 = {}".format(UNIFORM, cnt0, cnt1))
    train_set = [train_data, train_target]

    test_data = [data[i] for i in test_ind]
    test_target = [target[i] for i in test_ind]
    test_set = [test_data, test_target]
    return train_set, test_set, cnt0


def categorize_train_set1(examples):
    print("categorizing train set...")
    list_of_buckets = []
    for col in range(1, len(examples[0][0])):
        values = sorted({val[col] for val in examples[0]})
        min_val = min(values)
        max_val = max(values)
        len_of_interval = (max_val - min_val) / (NUM_OF_BUCKETS - 2)
        cur_col_buckets = []
        last_start = float('-inf')  # first min boundary
        for i in range(NUM_OF_BUCKETS - 1):
            start = last_start
            end = min_val + len_of_interval * i
            this_bucket = Bucket(start, end, i)
            cur_col_buckets.append(this_bucket)
            last_start = end
        this_bucket = Bucket(last_start, float('+inf'), NUM_OF_BUCKETS - 1)
        cur_col_buckets.append(this_bucket)
        list_of_buckets.append(cur_col_buckets)

        for i in range(len(examples[0])):
            if len_of_interval == 0:
                bucket_num = 0
            else:
                bucket_num = ceil((examples[0][i][col] - min_val) / len_of_interval)
            examples[0][i][col] = bucket_num
    return examples, list_of_buckets


def categorize_test_set1(test_set, buckets):
    print("categorizing test set...")
    for col in range(1, len(test_set[0][0])):
        for i in range(len(test_set[0])):
            for b in buckets[col - 1]:
                if b.start < test_set[0][i][col] <= b.end:
                    test_set[0][i][col] = b.label
                    break
    return test_set


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


def best_question(examples):
    n_features = len(examples[0][0]) - 1

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
    if len(examples[0]) < LEAF_SIZE:
        # print("    Max leaf size is reached!")
        return Leaf(examples)
    if cur_depth == MAX_DEPTH:
        print("    Max depth is reached!")
        return Leaf(examples)

    gain, question = best_question(examples)
    if gain == 0:
        return Leaf(examples)

    true_branch, false_branch = divide_data(examples, question)
    # if len(true_branch[0]) < LEAF_SIZE or len(false_branch[0]) < LEAF_SIZE:
    #     # print("Max leaf size is reached!")
    #     return Leaf(examples)

    true_branch = build_tree(true_branch, cur_depth + 1)
    false_branch = build_tree(false_branch, cur_depth + 1)
    return FeatureNode(question, true_branch, false_branch)


def printable_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}

    for lbl in counts.keys():
        probs[lbl] = str(round(counts[lbl] / total * 100)) + "%"
    return probs


def main_class(counts):
    # max_val = -1
    # main_cl = -1
    # print('count:', counts.items())
    # print(counts[0])
    # print(counts[1])
    # for k, v in counts.items():
    #     print("k: {}, v: {}".format(k, v))
    #     if v > max_val:
    #         max_val = v
    #         main_cl = k
    # return main_cl
    if len(counts) == 1:
        for k in counts.keys():
            return k
    val0 = counts[0]
    val1 = counts[1]
    coef = 1.0
    if UNIFORM:
        coef = 2.5
    if val1 > coef * val0:
        return 1
    else:
        return 0


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
        if target != main_class(node.predictions):
            node.errors += 1
        return node.predictions

    if node.question.match(dataset):
        return count_errors(dataset, target, node.true_branch)
    else:
        return count_errors(dataset, target, node.false_branch)


def count_and_prune(tree, train_set):
    for i in range(len(train_set[0])):
        count_errors(train_set[0][i], train_set[1][i], tree)
    prune(tree)


def prune(tree):
    z_a = 1.64

    if not isinstance(tree.true_branch, Leaf):
        tree.true_branch = prune(tree.true_branch)
    if not isinstance(tree.false_branch, Leaf):
        tree.false_branch = prune(tree.false_branch)

    if isinstance(tree.true_branch, Leaf) and isinstance(tree.false_branch, Leaf):
        true_size = sum(tree.true_branch.predictions.values())
        t_leaf_er = tree.true_branch.errors / true_size
        true_err = t_leaf_er + z_a * math.sqrt((t_leaf_er * (1 - t_leaf_er)) / true_size)

        false_size = sum(tree.false_branch.predictions.values())
        f_leaf_er = tree.false_branch.errors / false_size
        false_err = f_leaf_er + z_a * math.sqrt((f_leaf_er * (1 - f_leaf_er)) / false_size)

        total_size = true_size + false_size
        pes_err = (true_size * true_err + false_size * false_err) / total_size
        cur_err = (tree.true_branch.errors + tree.false_branch.errors) / total_size

        leafs_err = t_leaf_er + f_leaf_er

        print("cur err: {}, pes err: {}".format(cur_err, pes_err))
        if cur_err > pes_err:
            print("    !!! Do pruning !!!")
            return Leaf(None, tree.true_branch.predictions, tree.false_branch.predictions, leafs_err)
        else:
            return tree
    return tree


def main():
    data, target = load_tsv.load_pool(stop_size=10000)
    split_ratio = 0.8
    train_set, test_set, pivot = generate_sets(data, target, split_ratio, UNIFORM)
    train_size = int(len(data) * split_ratio)
    if UNIFORM:
        train_size = pivot * 2
    print("train size: {}, pivot: {}".format(train_size, pivot))

    print("modifying...")
    modified_train_set, buckets = categorize_train_set1(train_set)
    modified_test_set = categorize_test_set1(test_set, buckets)

    print("building tree...")
    root = build_tree(modified_train_set, 0)
    # print_tree(root)

    train_data = modified_train_set[0]
    train_target = modified_train_set[1]

    test_data = modified_test_set[0]
    test_target = modified_test_set[1]

    for i in range(len(modified_test_set[0])):
        count_errors(train_data[i], train_target[i], root)

    # print("pruning...")
    # count_and_prune(root, modified_train_set)

    all_classes = []
    print("\nclassifying...")
    for i in range(len(modified_test_set[0])):
        c = classify(test_data[i], root)
        # print("c:", c)
        all_classes.append(c)

    errors = 0
    tp = 1
    tn = 1
    fn = 1
    fp = 1
    rand_errors = 0
    for i in range(len(modified_test_set[0])):
        my_class = main_class(all_classes[i])
        rand_class = random.randint(0, train_size - 1)
        if rand_class >= pivot:
            rand_class = 1
        else:
            rand_class = 0

        if rand_class != test_target[i]:
            rand_errors += 1

        if my_class != test_target[i]:
            errors += 1

        if my_class == 1 and test_target[i] == 1:
            tp += 1

        if my_class == 0 and test_target[i] == 0:
            tn += 1

        if my_class == 0 and test_target[i] == 1:
            fn += 1

        if my_class == 1 and test_target[i] == 0:
            fp += 1
    print("tp: {}, fn: {}, fp: {}, tn: {}".format(tp-1, fn-1, fp-1, tn-1))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_measure = 2 * recall * precision / (recall + precision)
    print("errors: {} / {} => {}%,  F1: {},  R: {}, P: {}".
          format(errors, len(test_target), errors / len(test_target) * 100, f1_measure, recall, precision))
    print("random error: {} / {} => {}%".format(rand_errors, len(test_target), rand_errors / len(test_target) * 100))


start_time = time.time()
main()
end_time = time.time()
diff = end_time - start_time
print("~~~ %s sec ~~~" % diff)
print("~ {} min ~".format(diff / 60))
