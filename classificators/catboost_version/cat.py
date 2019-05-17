import random
import time
from catboost import CatBoostClassifier
import pandas as pd
from pprint import pprint as pp

UNIFORM = True


def load_pool(half_size=None, stop_size=None):
    df = pd.read_csv(r'C:\Users\Anastasiya\Desktop\диплом\sport_pool_20190305_20190307', delimiter='\t',
                     encoding='utf-8', nrows=60000, names=['query', 'factors', 'urls', 'target', 'clicks'],
                     usecols=['query', 'factors', 'urls', 'target'])
    cnt = 0
    data = []
    target = []
    cnt0 = 0
    cnt1 = 0
    len_of_facts = len(list(str(df.values[0][1])[8:].split()))

    last_query = ""  # for test
    for ex in df.values:
        query = str(ex[0])[6:]
        # if query.lower() == last_query.lower():  # for test
        #     continue  # for test
        last_query = query  # for test

        if query == "":
            continue

        facts = list(str(ex[1])[8:].split())
        targ = int(str(ex[3])[7:])

        if len(facts) != len_of_facts:
            continue

        cur_list = [query]
        cur_list += list(map(float, facts[:]))  # for test

        if half_size is not None:
            if cnt0 == half_size and cnt1 < half_size and targ == 0:
                continue
            if cnt1 == half_size and cnt0 < half_size and targ == 1:
                continue

        data.append(cur_list)
        if targ == 0:
            target.append(0)
            cnt0 += 1
        else:
            target.append(1)
            cnt1 += 1

        cnt += 1
        if cnt % 1000 == 0:
            print("total: {}, 0: {}, 1: {}".format(cnt, cnt0, cnt1))

        if stop_size is not None:
            if cnt >= stop_size:
                break

        if half_size is not None:
            if cnt0 >= half_size and cnt1 >= half_size:
                break
    print("data is loaded, cnt0 = {}, cnt1 = {}".format(cnt0, cnt1))
    return data, target


def generate_sets(data, target, split, uniform=False):
    train_size = int(len(data) * split)
    print("train set size: {}, test set size: {}".format(train_size, len(data) - train_size))
    # train_ind = sorted(random.sample(range(len(data)), train_size)) !!!
    train_ind = [i for i in range(0, train_size)]
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
    print("uniform: {}, in train set: cnt0 = {}, cnt1 = {}".format(uniform, cnt0, cnt1))
    train_set = [train_data, train_target]

    test_data = [data[i] for i in test_ind]
    test_target = [target[i] for i in test_ind]
    test_set = [test_data, test_target]

    return train_set, test_set, cnt0


def main():
    data, target = load_pool()

    for d in data:
        del(d[0])

    split_ratio = 0.8
    print("generating sets...")
    training_set, test_set, pivot = generate_sets(data, target, split_ratio, UNIFORM)

    train_data = training_set[0]
    train_labels = training_set[1]
    test_data = test_set[0]

    print("classifying...")
    model = CatBoostClassifier(iterations=5, class_weights=[4.395, 1.])
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)

    error_cnt = 0
    tp = 1
    tn = 1
    fn = 1
    fp = 1
    for p, t in zip(map(int, prediction), test_set[1]):
        if p != t:
            error_cnt += 1
        # print("p: {}, t: {}".format(p, t))
        if p == 1 and t == 1:
            tp += 1

        if p == 0 and t == 0:
            tn += 1

        if p == 0 and t == 1:
            fn += 1

        if p == 1 and t == 0:
            fp += 1

    print("tp: {}, fn: {}, fp: {}, tn: {}".format(tp - 1, fn - 1, fp - 1, tn - 1))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_measure = 2 * recall * precision / (recall + precision)
    print("errors: {} / {} => {}%,  F1: {},  R: {}, P: {}".
          format(error_cnt, len(test_data), error_cnt / len(test_data) * 100, f1_measure, recall, precision))


start_time = time.time()
main()
end_time = time.time()
print("time = {} sec".format(end_time - start_time))