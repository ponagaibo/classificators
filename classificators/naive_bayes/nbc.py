from math import ceil, log2
import pandas as pd
from pprint import pprint as pp
import random
import copy


headers = None


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


def load_tsv():
    df = pd.read_csv('little_data.tsv', delimiter='\t', encoding='utf-8')
    global headers
    headers = df.dtypes.index.values
    data_feature_names = headers[:-1]
    data = df[data_feature_names].values
    target_name = headers[-1]
    target = df[target_name].values
    return data, target


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


def create_buckets(values, num_of_buckets):
    feature_buckets = []
    whole_size = len(values)
    skip = 0
    last_start = float('-inf')
    denom = num_of_buckets
    for i in range(num_of_buckets):
        cur_amount = ceil(whole_size / denom)
        start = last_start
        end = values[skip + cur_amount - 1]
        if i == num_of_buckets - 1:
            end = float('+inf')
        last_start = end
        this_bucket = Bucket(start, end, i)
        this_bucket.add_to_cnt(cur_amount)
        feature_buckets.append(this_bucket)
        skip += cur_amount
        whole_size -= cur_amount
        denom -= 1
    return feature_buckets


def categorize_train_set(examples, num_of_buckets):
    new_set = copy.deepcopy(examples)
    list_of_buckets = []

    for col in range(1, len(new_set[0][0])):
        values = sorted({val[col] for val in new_set[0]})
        feature_buckets = create_buckets(values, num_of_buckets)
        list_of_buckets.append(feature_buckets)

        for i in range(len(new_set[0])):
            for j in feature_buckets:
                val = new_set[0][i][col]
                if j.start < val <= j.end:
                    new_set[0][i][col] = j.label
                    break
    return new_set, list_of_buckets


def categorize_test_set(test_set, buckets):
    new_test_set = copy.deepcopy(test_set)
    for col in range(1, len(buckets) + 1):
        for i in range(len(new_test_set[0])):
            val = new_test_set[0][i][col]
            for j in buckets[col - 1]:
                if j.start < val <= j.end:
                    new_test_set[0][i][col] = j.label
                    break
    return new_test_set


def get_elem(examples, class_name, feature_num=-1, feature_val=-1):
    cnt = 1
    if feature_num == -1 and feature_val == -1:
        for ex, t in zip(examples[0], examples[1]):
            if t == class_name:
                cnt += 1
                # print(ex)
    else:
        for ex, t in zip(examples[0], examples[1]):
            if ex[feature_num] == feature_val and t == class_name:
                cnt += 1
    return cnt


def get_elems_of_class(examples, class_name):
    for ex, t in zip(examples[0], examples[1]):
        if t == class_name:
            yield ex


def classify(train_set, test_set):
    print("classify this examples: ")
    pp(test_set)
    classes = []
    for ex in test_set[0]:
        max_ans = float('-inf')
        my_class = float('-inf')
        apriori_pr = find_class_freqs(train_set)
        train_len = len(train_set[0])
        for cls in apriori_pr.keys():
            apriori_pr[cls] /= train_len
        # print(apriori_pr)
        for cl in {0, 1}:
            sum = 0
            # pp(train_set)
            for col in range(1, len(test_set[0][0])):
                cur_val = ex[col]
                cnt_this_val_cl = get_elem(train_set, cl, col, cur_val)
                # print("col: " + str(col) + ", val: " + str(cur_val) + ", class: " + str(cl))
                # print(cnt_this_val_cl)
                cnt_this_cl = get_elem(train_set, cl)
                # print("in class " + str(cl) + " " + str(cnt_this_cl) + " elems")
                this_pr = cnt_this_val_cl / cnt_this_cl
                # print("this prob: " + str(this_pr) + ", log2: " + str(log2(this_pr)))
                sum += log2(this_pr)
            sum += log2(apriori_pr[cl])
            # print("sum = " + str(sum) + ", log2(pr(cl)): " + str(log2(apriori_pr[cl])))
            if sum >= max_ans:
                max_ans = sum
                my_class = cl
        classes.append(my_class)
        # print("for " + ex[0] + " class is " + str(my_class))
    for ex, cl, t in zip(test_set[0], classes, test_set[1]):
        print("for " + ex[0] + " predict: " + str(cl) + ", real: " + str(t))



def main():
    data, target = load_tsv()
    split_ratio = 0.7
    training_set, test_set = generate_sets(data, target, split_ratio)
    probs = find_class_freqs(training_set)
    print(probs)
    num_of_buckets = 4
    modified_train_set, buckets = categorize_train_set(training_set, num_of_buckets)

    modified_test_set = categorize_test_set(test_set, buckets)
    for i, j in zip(test_set[0], modified_test_set[0]):
        print(i)
        print(j)
        print()
    for b in buckets:
        print(b)
    print()
    print()
    classify(modified_train_set, modified_test_set)


main()
