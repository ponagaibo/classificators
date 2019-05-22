import random
import time
from catboost import CatBoostClassifier
import pandas as pd
from pprint import pprint as pp
import numpy as np


UNIFORM = False


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
    print(len_of_facts)

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


def generate_sets(data, target, split, uniform=False):
    train_size = int(len(data) * split)
    print("train set size: {}, test set size: {}".format(train_size, len(data) - train_size))
    train_ind = sorted(random.sample(range(len(data)), train_size))  # !!!
    # train_ind = [i for i in range(0, train_size)]
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


def print_importances(clf):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    for f in range(len(indices)):
        if indices[f] < 1110:
            continue
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# sport_pool_20190305_20190307
def load_pool2(file, half_size=None, stop_size=None, my_features=False):
    if file == "pool":
        filename = "\sport_pool_20190305_20190307"
    elif file == "common":
        filename = "\out_common"
    elif file == "allowed":
        filename = "\out_allowed"
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

    print("len:", len_of_facts)
    for ex in df.values:
        # pp(ex)
        query = str(ex[0])[6:]

        if query == "":
            continue

        facts = list(str(ex[1])[8:].split())
        urls = list(str(ex[2])[5:].split())
        targ = int(str(ex[3])[7:])
        clicks = list(str(ex[4])[7:].split())
        clicks = list(map(int, clicks))

        if len(facts) != len_of_facts:
            continue

        cur_list = [query]
        query_facts = facts[:]  #

        if my_features:
            # print("clicks:", clicks)
            num_of_clicks = sum(clicks)
            clicks5 = sum(clicks[:5])
            clicks10 = sum(clicks[:10])

            # print("sum:", num_of_clicks)
            f_hosts = open(r'C:\Users\Anastasiya\Desktop\диплом\project\classificators\parser\football_hosts.txt',
                           'r', encoding='utf-8')
            sport_hosts = f_hosts.read().split()
            f_hosts.close()

            hosts_to_check = {}
            cnt_ = 0
            top1 = False
            top3 = False
            top5 = False
            top7 = False
            top10 = False
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
                if host_with_two_dirs != "":
                    if hosts_to_check.get(host_with_two_dirs) is not None:
                        continue
                    # print("!   " + host_with_two_dirs + " in sport! place: " + str(cnt_))
                    hosts_to_check[host_with_two_dirs] = cnt_
                    if host_with_two_dirs in sport_hosts:
                        best_host = host_with_two_dirs

                if host_with_one_dir != "":
                    if hosts_to_check.get(host_with_one_dir) is not None:
                        continue
                    # print("!   " + host_with_one_dir + " in sport! place: " + str(cnt_))
                    hosts_to_check[host_with_one_dir] = cnt_
                    if host_with_one_dir in sport_hosts:
                        best_host = host_with_one_dir

                if hosts_to_check.get(host) is not None:
                    continue
                hosts_to_check[host] = cnt_
                if host in sport_hosts:
                    best_host = host
                if best_host != "":
                    sport_amount += 1
                    # print("!   " + best_host + " in sport! place: " + str(cnt_))
                    if cnt_ == 1:
                        top1 = True
                    if cnt_ <= 3:
                        top3 = True
                    if cnt_ <= 5:
                        top5 = True
                    if cnt_ <= 7:
                        top7 = True
                    if cnt_ <= 10:
                        top10 = True
                    if cnt_ <= 15:
                        top15 = True
            # print("top1: {}, top3: {}, top5: {}, top7: {}, top10: {}, top15: {}".
            #       format(str(top1), str(top3), str(top5), str(top7), str(top10), str(top15)))
            query_facts.append(int(top1))
            query_facts.append(int(top3))
            query_facts.append(int(top5))
            query_facts.append(int(top7))
            query_facts.append(int(top10))
            query_facts.append(int(top15))
            query_facts.append(num_of_clicks)
            query_facts.append(clicks5)
            query_facts.append(clicks10)  # на 3 месте
            query_facts.append(sport_amount)

        # print("new query facts:", query_facts)
        # print("len:", len(query_facts))
        cur_list += list(map(float, query_facts))
        # print(cur_list)

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


def main():
    my_f = True
    filename = "pool"
    data, target = load_pool2(filename, my_features=my_f, stop_size=60000)
    split_ratio = 0.8

    print("catboost")
    print("file:", filename)
    print("my features:", my_f)

    for d in data:
        del(d[0])

    training_set, test_set, pivot = generate_sets(data, target, split_ratio, UNIFORM)
    train_size = int(len(data) * split_ratio)
    if UNIFORM:
        train_size = pivot * 2
    print("train size: {}, pivot: {}".format(train_size, pivot))
    # pp(data[0])

    train_data = training_set[0]
    train_labels = training_set[1]
    test_data = test_set[0]

    print("classifying...")
    weight = 0.78
    print("weight of class 1:", weight)
    model = CatBoostClassifier(iterations=10, class_weights=[1, 4])
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)

    # print_importances(model)

    error_cnt = 0
    tp = 1
    tn = 1
    fn = 1
    fp = 1
    rand_errors = 0
    tp_rand = 1
    tn_rand = 1
    fn_rand = 1
    fp_rand = 1
    for p, t in zip(map(int, prediction), test_set[1]):
        rand_class = random.randint(0, train_size - 1)
        if rand_class >= pivot:
            rand_class = 1
        else:
            rand_class = 0

        if rand_class != t:
            rand_errors += 1

        if rand_class == 1 and t == 1:
            tp_rand += 1

        if rand_class == 0 and t == 0:
            tn_rand += 1

        if rand_class == 0 and t == 1:
            fn_rand += 1

        if rand_class == 1 and t == 0:
            fp_rand += 1

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
    print("errors: {} / {} => {}%,  F1: {},  R: {}, P: {}".format(error_cnt, len(test_data),
                                                                  round(error_cnt / len(test_data) * 100, 3),
                                                                  round(f1_measure, 3), round(recall, 3),
                                                                  round(precision, 3)))

    print("tp_r: {}, fn_r: {}, fp_r: {}, tn_r: {}".format(tp_rand - 1, fn_rand - 1, fp_rand - 1, tn_rand - 1))
    recall_rand = tp_rand / (tp_rand + fn_rand)
    precision_rand = tp_rand / (tp_rand + fp_rand)
    f1_measure_rand = 2 * recall_rand * precision_rand / (recall_rand + precision_rand)
    print("    rand error: {} / {} => {}%,  F1: {},  R: {}, P: {}".
          format(rand_errors, len(test_data), round(rand_errors / len(test_data) * 100, 3), round(f1_measure_rand, 3),
                 round(recall_rand, 3), round(precision_rand, 3)))


start_time = time.time()
main()
end_time = time.time()
print("time = {} sec".format(end_time - start_time))