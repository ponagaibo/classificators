import sys
import time
import pandas as pd
import numpy as np
import random
from pprint import pprint as pp
import math
from loader import load_tsv


MAX_DEPTH = 3
NUM_OF_BUCKETS = 8
LEAF_SIZE = 431

NUM_OF_TREES = 50
UNIFORM = False


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
    true_val, false_val = [], []
    true_targ, false_targ = [], []
    values = examples[0]
    targets = examples[1]
    for i in range(len(examples[0])):
        ex = values[i]
        if ex[question.column] >= question.value:
            true_val.append(ex)
            true_targ.append(targets[i])
        else:
            false_val.append(ex)
            false_targ.append(targets[i])
    true_val = [true_val, true_targ]
    false_val = [false_val, false_targ]
    return true_val, false_val


def entropy(examples):
    targets = examples[1]
    whole_size = len(examples[0])
    # only two classes - if 1 => in class, not in class otherwise
    in_class = sum(targets)
    not_in_class = whole_size - in_class
    if in_class == 0 or not_in_class == 0:
        return 0
    p_in = in_class / whole_size
    p_not_in = not_in_class / whole_size

    ent = -p_in * math.log2(p_in) - p_not_in * math.log2(p_not_in)
    return ent


def count_all_freqs(examples):
    n_features = len(examples[0][0]) - 1
    all_freqs = []
    for i in range(NUM_OF_BUCKETS):
        tmp_features = []
        for j in range(n_features + 1):
            tmp_classes = []
            for k in range(2):
                tmp_classes.append(0)
            tmp_features.append(tmp_classes)
        all_freqs.append(tmp_features)

    for ex, t in zip(examples[0], examples[1]):
        for col in range(1, n_features + 1):
            all_freqs[ex[col]][col][t] += 1  # bucket feature_num class 4x1033x2
    return all_freqs


def best_question(examples, rf=False):
    n_features = len(examples[0][0]) - 1
    feature_amount = math.sqrt(n_features) * 1.1
    all_freqs = count_all_freqs(examples)
    best_gain_ratio = 0
    best_q = None
    cur_entropy = entropy(examples)

    if rf:
        subfeatures_len = round(feature_amount)
        cur_features_num = sorted(random.sample([i for i in range(1, n_features + 1)], subfeatures_len))
    else:
        cur_features_num = [i for i in range(1, n_features + 1)]

    for col in cur_features_num:
        distinct_val = sorted(set(ex[col] for ex in examples[0]))
        for val in distinct_val:
            true0 = 0
            true1 = 0
            false0 = 0
            false1 = 0
            cnt_true = 0
            cnt_false = 0
            for i in range(0, NUM_OF_BUCKETS):
                if i >= val:
                    true0 += all_freqs[i][col][0]
                    true1 += all_freqs[i][col][1]
                    cnt_true += sum(all_freqs[i][col])
                else:
                    false0 += all_freqs[i][col][0]
                    false1 += all_freqs[i][col][1]
                    cnt_false += sum(all_freqs[i][col])

            if cnt_true == 0 or cnt_false == 0:
                continue

            # info_gain
            sum_ent = 0
            whole_size = float(len(examples[0]))

            len_t = float(cnt_true)
            # entropy
            if true1 == 0 or true0 == 0:
                e_true = 0
            else:
                pt_in = true1 / len_t
                pt_not_in = true0 / len_t
                e_true = -pt_in * math.log2(pt_in) - pt_not_in * math.log2(pt_not_in)
            # ---
            sum_ent += len_t * e_true / whole_size

            len_f = float(cnt_false)
            # entropy
            if false1 == 0 or false0 == 0:
                e_false = 0
            else:
                pf_in = false1 / len_f
                pf_not_in = false0 / len_f
                e_false = -pf_in * math.log2(pf_in) - pf_not_in * math.log2(pf_not_in)
            # ---

            sum_ent += len_f * e_false / whole_size
            gain = cur_entropy - sum_ent
            # ---

            # split_info
            true_ratio = len_t / whole_size
            false_ratio = len_f / whole_size
            split = -true_ratio * math.log2(true_ratio) - len_f / false_ratio * math.log2(false_ratio)
            # ---

            # gain_ratio
            # cur_gain_ratio = gain / split  # test gain instead gain ratio!!!
            cur_gain_ratio = gain
            # ---

            if cur_gain_ratio > best_gain_ratio:
                best_gain_ratio = cur_gain_ratio
                best_q = Question(col, val)
    return best_gain_ratio, best_q


def build_tree(examples, cur_depth, rf=False):
    if len(examples[0]) < LEAF_SIZE:
        # print("    Max leaf size is reached!")
        return Leaf(examples)
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


def categorize_train_set(examples):
    print("categorizing train set...")
    minimums = []
    lens = []
    minimums.append(0)
    lens.append(0)
    for col in range(1, len(examples[0][0])):
        values = sorted({val[col] for val in examples[0]})
        min_val = min(values)
        max_val = max(values)
        len_of_interval = (max_val - min_val) / (NUM_OF_BUCKETS - 2)
        minimums.append(min_val)
        lens.append(len_of_interval)

        for i in range(len(examples[0])):
            if len_of_interval == 0:
                bucket_num = 0
            else:
                bucket_num = math.ceil((examples[0][i][col] - min_val) / len_of_interval)
            examples[0][i][col] = bucket_num
    return examples, minimums, lens


def categorize_test_set(test_set, minimums, lens):
    print("categorizing test set...")
    for col in range(1, len(test_set[0][0])):
        for i in range(len(test_set[0])):
            if lens[col] == 0:
                bucket_num = 0
            else:
                bucket_num = math.ceil((test_set[0][i][col] - minimums[col]) / lens[col])
            if bucket_num < 0:
                bucket_num = 0
            if bucket_num >= NUM_OF_BUCKETS - 1:
                bucket_num = NUM_OF_BUCKETS - 1
            test_set[0][i][col] = bucket_num
    return test_set


def printable_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    # for kk, vv in counts.items():
    #     print(str(kk) + " : " + str(vv))
    for lbl in counts.keys():
        probs[lbl] = str(round(counts[lbl] / total * 100)) + "%"
    return probs


def main_class(counts, coef):
    if len(counts) == 1:
        for k in counts.keys():
            return k
    val0 = counts[0]
    val1 = counts[1]

    if val1 / (val0 + val1) >= coef:
        return 1
    else:
        return 0


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
    my_f = True
    filename = "shuffled"
    data, target = load_tsv.load_pool(filename, my_features=my_f, stop_size=5000)

    print("rf")
    print("file:", filename)
    print("my features:", my_f)
    print("MAX_DEPTH =", MAX_DEPTH)
    print("NUM_OF_BUCKETS =", NUM_OF_BUCKETS)
    print("LEAF_SIZE =", LEAF_SIZE)
    print("NUM_OF_TREES =", NUM_OF_TREES)

    split_ratio = 0.8
    train_set, test_set, pivot = generate_sets(data, target, split_ratio, UNIFORM)
    train_size = int(len(data) * split_ratio)
    if UNIFORM:
        train_size = pivot * 2
    print("train size: {}, pivot: {}".format(train_size, pivot))

    cnt0 = pivot
    cnt1 = train_size - cnt0
    class_fract = cnt1 / (cnt0 + cnt1)
    print("cnt1 / cnt0 = ", class_fract)

    print("modifying...")
    modified_train_set, minimums, lens = categorize_train_set(train_set)
    modified_test_set = categorize_test_set(test_set, minimums, lens)

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
        if i % 10 == 0:
            print("build tree {}/{}...".format(i, NUM_OF_TREES))
        root = build_tree(sub_train_set, 0, build_rf)
        list_of_trees.append(root)

    test_data = modified_test_set[0]
    test_target = modified_test_set[1]

    all_classes = []
    weight = class_fract  # при 0.22 ставит мало класса 1, при 0.6 чаще
    print("    weight =", weight)
    print("classifying...")
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
    tn = 1
    fn = 1
    fp = 1
    rand_errors = 0
    for i in range(len(modified_test_set[0])):
        my_class = main_class(all_classes[i], weight)
        rand_class = random.randint(0, train_size - 1)
        t = test_target[i]
        if rand_class >= pivot:
            rand_class = 1
        else:
            rand_class = 0

        if rand_class != t:
            rand_errors += 1

        if my_class != t:
            errors += 1

        if my_class == 1 and t == 1:
            tp += 1

        if my_class == 0 and t == 0:
            tn += 1

        if my_class == 0 and t == 1:
            fn += 1

        if my_class == 1 and t == 0:
            fp += 1

    print("tp: {}, fn: {}, fp: {}, tn: {}".format(tp-1, fn-1, fp-1, tn-1))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_measure = 2 * recall * precision / (recall + precision)
    print("    errors: {} / {} => {}%,  F1: {},  R: {}, P: {}"
          .format(errors, len(test_target), round(errors / len(test_target) * 100, 3),
                  round(f1_measure, 3), round(recall, 3), round(precision, 3)))
    print("    random error: {} / {} => {}%".format(rand_errors, len(test_target),
                                                    round(rand_errors / len(test_target) * 100, 3)))


start_time = time.time()
# orig_stdout = sys.stdout
# f = open(r'C:\Users\Anastasiya\Desktop\диплом\outs\rf_12_sq', 'w')
# sys.stdout = f
# for MAX_DEPTH in range(3, 11):
#     for NUM_OF_BUCKETS in range(4, 10, 2):
main()
# sys.stdout = orig_stdout
# f.close()
print("done rf 1.2")
end_time = time.time()
diff = end_time - start_time
print("~~~ %s sec ~~~" % diff)
print("~ {} min ~".format(diff / 60))

