import pandas as pd
import numpy as np
import random
from pprint import pprint as pp
from math import ceil, log2, sqrt


MAX_DEPTH = 5
NUM_OF_BUCKETS = 32
NUM_OF_TREES = 10

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


def load_pool():
    df = pd.read_csv(r'C:\Users\Anastasiya\Desktop\диплом\pool_tv_20190406', delimiter='\t', encoding='utf-8',
                     names=['factors', 'reqid', 'query', 'clicked'])
    cnt = 0
    data = []
    target = []
    cnt0 = 0
    cnt1 = 0
    half_size = 1900
    for ex in df.values:
        facts = list(str(ex[0])[8:].split())
        query = str(ex[2])[6:]
        if query == "":
            continue

        if len(facts) != 1097:
            continue

        clicked = str(ex[3])[8:]

        cur_list = [query]
        cur_list += list(map(float, facts[:]))

        if cnt0 == half_size and cnt1 < half_size and clicked == 'false':
            continue
        if cnt1 == half_size and cnt0 < half_size and clicked == 'true':
            continue

        data.append(cur_list)
        if clicked == 'false':
            target.append(0)
            cnt0 += 1
        else:
            target.append(1)
            cnt1 += 1
        cnt += 1
        if cnt % 1000 == 0:
            print("total: {}, 0: {}, 1: {}".format(cnt, cnt0, cnt1))

        # if cnt >= 500:
        #     break

        if cnt0 >= half_size and cnt1 >= half_size:
            break
    print("data is loaded")
    return data, target


def load_tsv():
    df = pd.read_csv('little_data.tsv', delimiter='\t', encoding='utf-8')
    headers = df.dtypes.index.values
    data_feature_names = headers[:-1]
    data = df[data_feature_names].values
    target_name = headers[-1]
    target = df[target_name].values
    return data, target


def generate_sets(data, target, split_coef):
    train_size = int(len(data) * split_coef)
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

    ent = -p_in * log2(p_in) - p_not_in * log2(p_not_in)
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
    sum_info = len_t / len_whole * log2(len_t / len_whole) + len_f / len_whole * log2(len_f / len_whole)
    return -sum_info


def gain_ratio(true_branch, false_branch, whole_entropy):
    gain = info_gain(true_branch, false_branch, whole_entropy)
    split = split_info(true_branch, false_branch)
    return gain / split


def best_question(examples, rf=False):
    n_features = len(examples[0][0]) - 1
    feature_amount = sqrt(n_features)

    best_gain_ratio = 0
    best_q = None
    cur_entropy = entropy(examples)  # ???

    if rf:
        subfeatures_len = round(feature_amount)
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


def main():
    # for every tree choose subset and build three
    # when trees are built, use test set to everyone and collect answers
    # most popular answer is a global answer

    data, target = load_pool()
    split_ratio = 0.8
    train_set, test_set = generate_sets(data, target, split_ratio)

    print("modifying...")
    modified_train_set, buckets = categorize_train_set1(train_set)
    modified_test_set = categorize_test_set1(test_set, buckets)

    list_of_trees = []
    for i in range(NUM_OF_TREES):
        build_rf = True
        sub_data = []
        sub_target = []
        for _ in modified_train_set[0]:
            k = random.randrange(0, len(modified_train_set[0]))
            sub_data.append(modified_train_set[0][k])
            sub_target.append(modified_train_set[1][k])
        sub_train_set = [sub_data, sub_target]

        print("build tree {}/{}...".format(i, NUM_OF_TREES))
        root = build_tree(sub_train_set, 0, build_rf)
        list_of_trees.append(root)

    test_data = modified_test_set[0]
    test_target = modified_test_set[1]

    print("classifying...")
    all_classes = []
    for i in range(len(modified_test_set[0])):
        total_classes = {}
        tree_cnt = 1
        for tree in list_of_trees:
            c = classify(test_data[i], tree)
            for kk, vv in c.items():
                if kk not in total_classes:
                    total_classes[kk] = 0
                total_classes[kk] += vv
            tree_cnt += 1
        all_classes.append(total_classes)

    errors = 0
    tp = 1
    fn = 1
    fp = 1
    for i in range(len(modified_test_set[0])):
        my_class = main_class(all_classes[i])
        if my_class != test_target[i]:
            errors += 1

        if my_class == 1 and test_target[i] == 1:
            tp += 1

        if my_class == 0 and test_target[i] == 1:
            fn += 1

        if my_class == 1 and test_target[i] == 0:
            fp += 1

    print("tp: {}, fn: {}, fp: {}".format(tp, fn, fp))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_measure = 2 * recall * precision / (recall + precision)
    print("errors: {} / {} => {}%,  F1: {},  R: {}, P: {}".
          format(errors, len(test_target), errors / len(test_target) * 100, f1_measure, recall, precision))


main()
