from math import ceil, log2
import pandas as pd
from pprint import pprint as pp
import random
import copy
import time


NUM_OF_BUCKETS = 32


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


def generate_sets(data, target, split):
    train_size = int(len(data) * split)
    train_ind = sorted(random.sample(range(len(data)), train_size))
    test_ind = [i for i in range(len(data)) if i not in train_ind]

    train_data = [data[i] for i in train_ind]
    train_target = [target[i] for i in train_ind]
    train_set = [train_data, train_target]

    test_data = [data[i] for i in test_ind]
    test_target = [target[i] for i in test_ind]
    test_set = [test_data, test_target]

    return train_set, test_set


def find_class_freqs(training_set):
    target = training_set[1]
    probs = {}
    for t in target:
        if t not in probs:
            probs[t] = 0
        probs[t] += 1
    return probs


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


def get_freqs(examples):
    freqs = {}
    for ex, t in zip(examples[0], examples[1]):
        if t not in freqs:
            freqs[t] = 1
        freqs[t] += 1
    return freqs


def get_value_freqs(examples):
    val_freqs = []
    freqs0 = []
    freqs1 = []
    freqs0.append({})
    freqs1.append({})
    for col in range(1, len(examples[0][0])):
        column_freqs0 = {}
        column_freqs1 = {}
        for val in range(0, NUM_OF_BUCKETS):
            column_freqs0[val] = 1
            column_freqs1[val] = 1

        for ex, t in zip(examples[0], examples[1]):
            if t == 0:
                column_freqs0[ex[col]] += 1
            else:
                column_freqs1[ex[col]] += 1
        freqs0.append(column_freqs0)
        freqs1.append(column_freqs1)
    val_freqs.append(freqs0)
    val_freqs.append(freqs1)
    return val_freqs


def classify(train_set, test_set):
    ex_cnt = 0
    errors = 0
    cnt0 = 0
    cnt1 = 0
    real0 = 0
    real1 = 0

    tp = 1
    fn = 1
    fp = 1

    print("classifying...")
    classes = []
    apriori_pr = find_class_freqs(train_set)
    train_len = len(train_set[0])
    for cls in apriori_pr.keys():
        apriori_pr[cls] /= train_len

    class_freqs = get_freqs(train_set)
    val_freqs = get_value_freqs(train_set)

    for ex, t in zip(test_set[0], test_set[1]):
        max_ans = float('-inf')
        my_class = float('-inf')

        for cl in {0, 1}:
            cur_sum = 0
            cnt_this_cl = class_freqs[cl]

            for col in range(1, len(test_set[0][0])):
                cur_val = ex[col]
                this_pr = val_freqs[cl][col][cur_val] / cnt_this_cl
                cur_sum += log2(this_pr)
            cur_sum += log2(apriori_pr[cl])
            if cur_sum >= max_ans:
                max_ans = cur_sum
                my_class = cl
        classes.append(my_class)
        if my_class != t:
            errors += 1

        if my_class == 0:
            cnt0 += 1
        else:
            cnt1 += 1

        if t == 0:
            real0 += 1
        else:
            real1 += 1

        if my_class == 1 and t == 1:
            tp += 1

        if my_class == 0 and t == 1:
            fn += 1

        if my_class == 1 and t == 0:
            fp += 1

        ex_cnt += 1
        if ex_cnt % 100 == 0:
            print("cnt:", ex_cnt)

    print("tp: {}, fn: {}, fp: {}".format(tp, fn, fp))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_measure = 2 * recall * precision / (recall + precision)
    print("errors: {} / {} => {}%,  F1: {},  R: {}, P: {}"
          .format(errors, len(classes), errors / len(classes) * 100, f1_measure, recall, precision))


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

        # if cnt >= 1000:
        #     break

        if cnt0 >= half_size and cnt1 >= half_size:
            break
    print("data is loaded")

    return data, target


def main():
    data, target = load_pool()
    split_ratio = 0.8
    training_set, test_set = generate_sets(data, target, split_ratio)
    modified_train_set, buckets = categorize_train_set1(training_set)
    modified_test_set = categorize_test_set1(test_set, buckets)
    print("modified")

    start = time.time()
    classify(modified_train_set, modified_test_set)
    end = time.time()
    print("time =", end - start)


main()
