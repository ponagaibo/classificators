from math import ceil
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


def find_a_priori_prob(training_set):
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


def categorize(examples, num_of_buckets):
    new_set = copy.deepcopy(examples)
    list_of_buckets = []

    for k in range(1, len(new_set[0][0])):
        feature_buckets = []
        values = sorted({val[k] for val in new_set[0]})
        list_of_buckets.append(create_buckets(values, num_of_buckets))

        for i in range(len(new_set[0])):
            for j in feature_buckets:
                val = new_set[0][i][k]
                if j.start < val <= j.end:
                    new_set[0][i][k] = j.label
                    break
    return new_set, list_of_buckets


def main():
    data, target = load_tsv()
    split_ratio = 0.7
    training_set, test_set = generate_sets(data, target, split_ratio)
    # pp(training_set)
    probs = find_a_priori_prob(training_set)
    print(probs)
    num_of_buckets = 4
    modified_train_set, buckets = categorize(training_set, num_of_buckets)
    pp(training_set[0])
    for i, j in zip(training_set[0], modified_train_set[0]):
        print(i)
        print(j)
        print()
    for b in buckets:
        print(b)


main()
