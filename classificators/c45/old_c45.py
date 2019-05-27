import pandas as pd
from math import ceil
import numpy as np
import random
from pprint import pprint as pp
import math


MAX_DEPTH = 40
NUM_OF_BUCKETS = 4
LEAF_SIZE = 501  # 101


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


def load_pool(file, half_size=None, stop_size=None, my_features=False):
    if file == "pool":
        filename = "\sport_pool_20190305_20190307"
    elif file == "common":
        filename = "\out_common"
    elif file == "allowed":
        filename = "\out_allowed"
    elif file == "shuffled":
        filename = "\shuf"
    else:
        print("filename error")

    df = pd.read_csv(r'C:\Users\Anastasiya\Desktop\диплом' + filename, delimiter='\t',
                     encoding='utf-8', nrows=300000, low_memory=False,
                     names=['query', 'factors', 'urls', 'target', 'clicks'])

    cnt = 0
    data = []
    target = []
    cnt0 = 0
    cnt1 = 0
    len_of_facts = len(list(str(df.values[0][1])[8:].split()))

    # print("len:", len_of_facts)
    for ex in df.values:
        query = str(ex[0])[6:]

        if query == "":
            continue

        facts = list(str(ex[1])[8:].split())
        urls = list(str(ex[2])[5:].split())
        targ = int(str(ex[3])[7:])
        clicks = list(str(ex[4])[7:].split())
        clicks = list(map(int, clicks))
        len_c = len(clicks)
        if len_c == 0:
            continue
        # if len_c != 10 and len_c != 15:
        #     print("{}: len of clicks: {}".format(cnt, len(clicks)))

        if len(facts) != len_of_facts:
            continue

        cur_list = [query]
        query_facts = facts[:]  #

        if my_features:
            # print("clicks:", clicks)
            num_of_clicks = sum(clicks)
            clicks1 = clicks[0]
            clicks2 = sum(clicks[:2])
            clicks3 = sum(clicks[:3])
            clicks4 = sum(clicks[:4])
            clicks5 = sum(clicks[:5])
            clicks6 = sum(clicks[:6])
            clicks7 = sum(clicks[:7])
            clicks8 = sum(clicks[:8])
            clicks9 = sum(clicks[:9])
            clicks10 = sum(clicks[:10])
            clicks11 = sum(clicks[:11])
            clicks12 = sum(clicks[:12])
            clicks13 = sum(clicks[:13])
            clicks14 = sum(clicks[:14])
            clicks15 = sum(clicks[:15])
            num_clicked_sport = 0

            # print("sum:", num_of_clicks)
            f_hosts = open(r'C:\Users\Anastasiya\Desktop\диплом\project\classificators\parser\football_hosts.txt',
                           'r', encoding='utf-8')
            sport_hosts = f_hosts.read().split()
            f_hosts.close()

            hosts_to_check = {}
            cnt_ = 0
            top1 = False
            top2 = False
            top3 = False
            top4 = False
            top5 = False
            top6 = False
            top7 = False
            top8 = False
            top9 = False
            top10 = False
            top11 = False
            top12 = False
            top13 = False
            top14 = False
            top15 = False
            sport_amount = 0
            for u in urls:
                whole_url = u.split("/")
                whole_url = list(filter(None, whole_url))
                host = whole_url[1]
                list_of_host_words = host.split(".")
                if list_of_host_words[0] == "www":
                    del (list_of_host_words[0])

                host = ".".join(list_of_host_words)
                host_with_one_dir = ""
                if len(whole_url) >= 3:
                    directory = whole_url[2]
                    host_with_one_dir = host + "/" + directory

                host_with_two_dirs = ""
                if len(whole_url) >= 4:
                    host_with_two_dirs = host_with_one_dir + "/" + whole_url[3]

                if (hosts_to_check.get(host_with_two_dirs) is None
                        and hosts_to_check.get(host_with_one_dir) is None
                        and len(whole_url) >= 3):
                    cnt_ += 1

                best_host = ""
                place_of_best_host = 0
                if host_with_two_dirs != "":
                    if hosts_to_check.get(host_with_two_dirs) is not None:
                        continue
                    # print("!   " + host_with_two_dirs + " in sport! place: " + str(cnt_))
                    hosts_to_check[host_with_two_dirs] = cnt_
                    if host_with_two_dirs in sport_hosts:
                        best_host = host_with_two_dirs
                        place_of_best_host = cnt_

                if host_with_one_dir != "":
                    if hosts_to_check.get(host_with_one_dir) is not None:
                        continue
                    # print("!   " + host_with_one_dir + " in sport! place: " + str(cnt_))
                    hosts_to_check[host_with_one_dir] = cnt_
                    if host_with_one_dir in sport_hosts:
                        best_host = host_with_one_dir
                        place_of_best_host = cnt_

                if hosts_to_check.get(host) is not None:
                    continue
                hosts_to_check[host] = cnt_
                if host in sport_hosts:
                    best_host = host
                    place_of_best_host = cnt_
                if best_host != "":
                    if clicks[place_of_best_host-1] == 1:
                        num_clicked_sport += 1
                    sport_amount += 1
                    # print("!   " + best_host + " in sport! place: " + str(cnt_))
                    if cnt_ == 1:
                        top1 = True
                    if cnt_ <= 2:
                        top2 = True
                    if cnt_ <= 3:
                        top3 = True
                    if cnt_ <= 4:
                        top4 = True
                    if cnt_ <= 5:
                        top5 = True
                    if cnt_ <= 6:
                        top6 = True
                    if cnt_ <= 7:
                        top7 = True
                    if cnt_ <= 8:
                        top8 = True
                    if cnt_ <= 9:
                        top9 = True
                    if cnt_ <= 10:
                        top10 = True
                    if cnt_ <= 11:
                        top11 = True
                    if cnt_ <= 12:
                        top12 = True
                    if cnt_ <= 13:
                        top13 = True
                    if cnt_ <= 14:
                        top14 = True
                    if cnt_ <= 15:
                        top15 = True
            # print("top1: {}, top3: {}, top5: {}, top7: {}, top10: {}, top15: {}".
            #       format(str(top1), str(top3), str(top5), str(top7), str(top10), str(top15)))

            query_facts.append(int(top1))
            query_facts.append(int(top2))
            query_facts.append(int(top3))
            query_facts.append(int(top4))
            query_facts.append(int(top5))
            query_facts.append(int(top6))
            query_facts.append(int(top7))
            query_facts.append(int(top8))
            query_facts.append(int(top9))
            query_facts.append(int(top10))
            query_facts.append(int(top11))
            query_facts.append(int(top12))
            query_facts.append(int(top13))
            query_facts.append(int(top14))
            query_facts.append(int(top15))

            query_facts.append(num_of_clicks)

            query_facts.append(clicks1)
            query_facts.append(clicks2)
            query_facts.append(clicks3)
            query_facts.append(clicks4)
            query_facts.append(clicks5)
            query_facts.append(clicks6)
            query_facts.append(clicks7)
            query_facts.append(clicks8)
            query_facts.append(clicks9)
            query_facts.append(clicks10)  # на 3 месте
            query_facts.append(clicks11)
            query_facts.append(clicks12)
            query_facts.append(clicks13)
            query_facts.append(clicks14)
            query_facts.append(clicks15)

            query_facts.append(sport_amount)
            query_facts.append(num_clicked_sport)

        # print("new query facts:", query_facts)
        cur_list += list(map(float, query_facts))

        if half_size is not None:
            if cnt0 == half_size and cnt1 < half_size and targ == 0:
                continue
            if cnt1 == half_size and cnt0 < half_size and targ == 1:
                continue

        data.append(cur_list)
        if targ == 0:
            target.append(0)  # not inverted!!!!
            cnt0 += 1
        else:
            target.append(1)
            cnt1 += 1

        cnt += 1
        if cnt % 5000 == 0:
            print("total: {}, 0: {}, 1: {}".format(cnt, cnt0, cnt1))

        if stop_size is not None:
            if cnt >= stop_size:
                break

        if half_size is not None:
            if cnt0 >= half_size and cnt1 >= half_size:
                break
    print("data is loaded, cnt0 = {}, cnt1 = {}".format(cnt0, cnt1))
    return data, target


def load_tsv():
    df = pd.read_csv('little_data.tsv', delimiter='\t', encoding='utf-8')
    headers = df.dtypes.index.values
    data_feature_names = headers[:-1]
    data = df[data_feature_names].values
    target_name = headers[-1]
    target = df[target_name].values
    print("headers:")
    print(headers)
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
    return gain  # !!!


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


def best_question1(examples):
    n_features = len(examples[0][0]) - 1
    all_freqs = count_all_freqs(examples)
    best_gain_ratio = 0
    best_q = None
    cur_entropy = entropy(examples)

    for col in range(1, n_features + 1):
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

            # leaf size
            # if cnt_true < LEAF_SIZE or cnt_false < LEAF_SIZE:
            #     continue

            # info_gain
            sum_ent = 0
            whole_size = float(len(examples[0]))

            len_t = float(cnt_true)
            # entropy
            if true1 == 0 or true0 == 0:
                continue
            pt_in = true1 / whole_size
            pt_not_in = true0 / whole_size
            e_true = -pt_in * math.log2(pt_in) - pt_not_in * math.log2(pt_not_in)
            # ---
            sum_ent += len_t * e_true / whole_size

            len_f = float(cnt_false)
            # entropy
            if false1 == 0 or false0 == 0:
                continue
            pf_in = false1 / whole_size
            pf_not_in = false0 / whole_size
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
            # cur_gain_ratio = gain / split  # test gain instead gainratio!!!
            cur_gain_ratio = gain
            # ---

            if cur_gain_ratio > best_gain_ratio:
                best_gain_ratio = cur_gain_ratio
                best_q = Question(col, val)
    return best_gain_ratio, best_q


def best_question(examples):
    n_features = len(examples[0][0]) - 1

    best_gain_ratio = 0
    best_q = None
    cur_entropy = entropy(examples)  # ???
    targets = examples[1]
    whole_size = len(examples[0])
    for col in range(1, n_features + 1):
        distinct_val = sorted(set(ex[col] for ex in examples[0]))
        for val in distinct_val:
            # divide_data start
            true_branch, false_branch = [], []
            true_targets, false_targets = [], []
            for i in range(len(examples[0])):
                ex = examples[0][i]
                if ex[col] >= val:
                    true_branch.append(ex)
                    true_targets.append(targets[i])
                else:
                    false_branch.append(ex)
                    false_targets.append(targets[i])
            true_branch = [true_branch, true_targets]
            false_branch = [false_branch, false_targets]
            # divide_data end
            # true_branch, false_branch = divide_data(examples, Question(col, val))

            if len(true_branch[0]) == 0 or len(false_branch[0]) == 0:
                continue

            # info_gain start
            sum_ent = 0

            true_b_size = len(true_branch[0])

            # entropy start
            in_class_t = sum(true_targets)
            not_in_class_t = true_b_size - in_class_t

            if not_in_class_t == 0 or in_class_t == 0:
                true_e = 0
            else:
                p_in_t = in_class_t / true_b_size
                p_not_in_t = not_in_class_t / true_b_size
                true_e = -p_in_t * math.log2(p_in_t) - p_not_in_t * math.log2(p_not_in_t)
            # entropy end

            sum_ent += true_b_size * true_e / whole_size

            false_b_size = len(false_branch[0])

            # entropy start
            in_class_f = sum(false_targets)
            not_in_class_f = false_b_size - in_class_f
            if not_in_class_f == 0 or in_class_f == 0:
                false_e = 0
            else:
                p_in_f = in_class_f / false_b_size
                p_not_in_f = not_in_class_f / false_b_size
                false_e = -p_in_f * math.log2(p_in_f) - p_not_in_f * math.log2(p_not_in_f)
            # entropy end

            sum_ent += false_b_size * false_e / whole_size

            cur_gain_ratio = cur_entropy - sum_ent
            # info_gain end
            # cur_gain_ratio = gain_ratio(true_branch, false_branch, cur_entropy)

            if cur_gain_ratio > best_gain_ratio:
                best_gain_ratio = cur_gain_ratio
                best_q = Question(col, val)
    return best_gain_ratio, best_q


# original
def best_question0(examples):
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
        print("Max depth is reached!")
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
        if target != main_class(node.predictions, 1):
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

        if cur_err > pes_err:
            print("Do pruning!")
            return Leaf(None, tree.true_branch.predictions, tree.false_branch.predictions, leafs_err)
        else:
            return tree
    return tree


def main():
    data, target = load_pool("shuffled", stop_size=3000)
    split_ratio = 0.8
    train_set, test_set = generate_sets(data, target, split_ratio)

    print("modifying...")
    modified_train_set, buckets = categorize_train_set1(train_set)
    modified_test_set = categorize_test_set1(test_set, buckets)

    print("building tree...")
    root = build_tree(modified_train_set, 0)

    train_data = modified_train_set[0]
    train_target = modified_train_set[1]

    test_data = modified_test_set[0]
    test_target = modified_test_set[1]

    # for i in range(len(modified_test_set[0])):
    #     count_errors(train_data[i], train_target[i], root)
    #
    # print("pruning...")
    # count_and_prune(root, modified_train_set)

    all_classes = []
    print("classifying...")
    for i in range(len(modified_test_set[0])):
        c = classify(test_data[i], root)
        all_classes.append(c)

    errors = 0
    tp = 1
    fn = 1
    fp = 1
    for i in range(len(modified_test_set[0])):
        my_class = main_class(all_classes[i], 0.32)
        if my_class != test_target[i]:
            errors += 1

        if my_class == 1 and test_target[i] == 1:
            tp += 1

        if my_class == 0 and test_target[i] == 1:
            fn += 1

        if my_class == 1 and test_target[i] == 0:
            fp += 1
    print("tp: {}, fn: {}, fp: {}".format(tp-1, fn-1, fp-1))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_measure = 2 * recall * precision / (recall + precision)
    print("errors: {} / {} => {}%,  F1: {},  R: {}, P: {}".
          format(errors, len(test_target), errors / len(test_target) * 100, f1_measure, recall, precision))


main()
