from math import ceil, log2
import pandas as pd
from pprint import pprint as pp
import random
import copy
import time
from loader import load_tsv
import cProfile
import sys


NUM_OF_BUCKETS = 400
UNIFORM = False


def generate_sets(data, target, split, uniform=False):
    train_size = int(len(data) * split)
    print("    train set size: {}, test set size: {}".format(train_size, len(data) - train_size))
    train_ind = sorted(random.sample(range(len(data)), train_size))  # !!!
    # train_ind = [i for i in range(0, train_size)]
    test_ind = [i for i in range(len(data)) if i not in train_ind]

    cnt0 = 0
    cnt1 = 0
    train_data = [data[i] for i in train_ind]
    train_target = []
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
    print("    uniform: {}, in train set: cnt0 = {}, cnt1 = {}".format(uniform, cnt0, cnt1))
    train_set = [train_data, train_target]

    test_data = [data[i] for i in test_ind]
    test_target = [target[i] for i in test_ind]
    test_set = [test_data, test_target]

    return train_set, test_set, cnt0


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
                bucket_num = ceil((examples[0][i][col] - min_val) / len_of_interval)

            examples[0][i][col] = bucket_num
    return examples, minimums, lens


def categorize_test_set(test_set, minimums, lens):
    print("categorizing test set...")
    for col in range(1, len(test_set[0][0])):
        for i in range(len(test_set[0])):
            if lens[col] == 0:
                bucket_num = 0
            else:
                bucket_num = ceil((test_set[0][i][col] - minimums[col]) / lens[col])
            if bucket_num < 0:
                bucket_num = 0
            if bucket_num >= NUM_OF_BUCKETS - 1:
                bucket_num = NUM_OF_BUCKETS - 1
            test_set[0][i][col] = bucket_num
    return test_set


def get_class_freqs(examples):
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


def classify(train_set, test_set, fract, pivot, train_size):
    errors = 0
    rand_errors = 0

    tp = 1
    fn = 1
    fp = 1
    tn = 1

    train_len = len(train_set[0])

    class_freqs = get_class_freqs(train_set)
    val_freqs = get_value_freqs(train_set)

    print("    adding weights to class...")
    class_freqs[1] = int(fract * class_freqs[1])
    print("    adding weights to values in class...")
    for line in val_freqs[1]:
        for key in line:
            line[key] = int(fract * line[key])

    weight = 1.05
    # weight = 1.05
    print("    weight:", weight)
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
            cur_sum += log2((cnt_this_cl - 1) / train_len)

            if weight * cur_sum >= max_ans:  # add weight to increase precision
                max_ans = cur_sum
                my_class = cl

        rand_class = random.randint(0, train_size - 1)
        if rand_class >= pivot:
            rand_class = 1
        else:
            rand_class = 0
        # rand_class = random.randint(0, 1)  # make 50/50

        if rand_class != t:
            rand_errors += 1

        if my_class != t:
            errors += 1

        if my_class == 1 and t == 1:
            tp += 1

        if my_class == 0 and t == 1:
            fn += 1

        if my_class == 1 and t == 0:
            fp += 1

        if my_class == 0 and t == 0:
            tn += 1

    print("    tp: {}, fn: {}, fp: {}, tn: {}, 0: {}, 1: {}".format(tp-1, fn-1, fp-1, tn-1, tn + fn - 2, tp + fp - 2))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_measure = 2 * recall * precision / (recall + precision)
    print("    errors: {} / {} => {}%,  F1: {},  R: {}, P: {}"
          .format(errors, len(test_set[0]), round(errors / len(test_set[0]) * 100, 3),
                  round(f1_measure, 3), round(recall, 3), round(precision, 3)))
    print("    random error: {} / {} => {}%".format(rand_errors, len(test_set[0]),
                                                    round(rand_errors / len(test_set[0]) * 100, 3)))


def main():
    my_f = False
    filename = "shuffled"
    data, target = load_tsv.load_pool(file=filename, my_features=True, stop_size=50000)
    # pp([data, target])
    print("nbc")
    print("file:", filename)
    print("my features:", my_f)
    print("NUM_OF_BUCKETS =", NUM_OF_BUCKETS)
    split_ratio = 0.8
    print("generating sets...")
    train_set, test_set, pivot = generate_sets(data, target, split_ratio, UNIFORM)

    train_size = int(len(data) * split_ratio)
    print("pivot: {}, train size: {}".format(pivot, train_size))
    cnt0 = pivot
    cnt1 = train_size - cnt0
    class_fract = cnt0 / cnt1
    print("cnt0 / cnt1 = ", class_fract)

    print("modifying...")
    modified_train_set, minimums, lens = categorize_train_set(train_set)
    modified_test_set = categorize_test_set(test_set, minimums, lens)

    print("classifying...")
    classify(modified_train_set, modified_test_set, class_fract, pivot, train_size)


start_time = time.time()
# pr = cProfile.Profile()
# pr.enable()
orig_stdout = sys.stdout
f = open(r'C:\Users\Anastasiya\Desktop\диплом\outs\nbc_res_without', 'w')
sys.stdout = f
for NUM_OF_BUCKETS in range(10, 420, 10):
    main()
sys.stdout = orig_stdout
f.close()
print("done without my f")
# pr.disable()
end_time = time.time()
diff = end_time - start_time
print("~~~ %s sec ~~~" % diff)
print("~ {} min ~".format(diff / 60))
# pr.print_stats(sort="cumtime")
