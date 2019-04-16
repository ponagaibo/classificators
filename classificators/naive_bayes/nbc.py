from math import ceil, log2
import pandas as pd
from pprint import pprint as pp
import random
import copy
import time


headers = None
num_of_buckets = 80


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
    print('headers:', headers)
    data_feature_names = headers[:-1]
    data = df[data_feature_names].values
    print("data:")
    print(data)
    target_name = headers[-1]
    target = df[target_name].values
    print('target:')
    print(target)
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

    # print("\ntrain_set:")
    # for tr in enumerate(zip(train_set[0], train_set[1])):
    #     print(tr)
    # print(train_set)
    # print("test_set:")
    # for ts in enumerate(test_set[0]):
    #     print(ts)
    # print(test_set)
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
    whole_size = len(values) # amount of examples
    skip = 0
    last_start = float('-inf') # first min boundary
    denom = num_of_buckets
    for i in range(num_of_buckets):
        cur_amount = ceil(whole_size / denom) # average amount of examples in bucket
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
    print("catecorizing...")
    new_set = copy.deepcopy(examples)
    list_of_buckets = []
    print("len: " + str(len(new_set[0][0]))) # feature num
    print("new_set[0]:")
    pp(new_set[0])
    for col in range(1, len(new_set[0][0])):
        values = sorted({val[col] for val in new_set[0]}) # iterating through examples
        print("values:", values)
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
    print("categorize test set")
    pp(test_set)
    new_test_set = copy.deepcopy(test_set)
    for col in range(1, len(buckets) + 1):
        for i in range(len(new_test_set[0])):
            val = new_test_set[0][i][col]
            print("col: {}, i: {}, val: {}".format(col, i, val))
            for j in buckets[col - 1]:
                if j.start < val <= j.end:
                    new_test_set[0][i][col] = j.label
                    print("set bucket:", j.label)
                    break
    print("new test set:")
    pp(new_test_set)
    return new_test_set


def categorize_train_set1(examples, num_of_buckets):
    print("categorizing train set...")
    list_of_buckets = []
    # print("len of ex", len(examples[0][0]))
    for col in range(1, len(examples[0][0])):
        values = sorted({val[col] for val in examples[0]})
        # print("value len: {}, col: {}".format(len(values), col))
        # print(values)
        min_val = min(values)
        # print("min:", min_val)
        max_val = max(values)
        # print("max:", max_val)
        len_of_interval = (max_val - min_val) / (num_of_buckets - 2)
        # print("len of interval:", len_of_interval)
        cur_col_buckets = []
        last_start = float('-inf')  # first min boundary
        for i in range(num_of_buckets - 1):
            start = last_start
            end = min_val + len_of_interval * i
            this_bucket = Bucket(start, end, i)
            cur_col_buckets.append(this_bucket)
            # print("#{}: ({} : {}]".format(i, start, end))
            last_start = end
        this_bucket = Bucket(last_start, float('+inf'), num_of_buckets - 1)
        cur_col_buckets.append(this_bucket)
        # print("#{}: ({} : {}]".format(num_of_buckets - 1, last_start, float('+inf')))
        list_of_buckets.append(cur_col_buckets)

        for i in range(len(examples[0])):
            # print("cur val:", examples[0][i][col])
            if len_of_interval == 0:
                bucket_num = 0
            else:
                bucket_num = ceil((examples[0][i][col] - min_val) / len_of_interval)
            # print("bucket number:", bucket_num)
            examples[0][i][col] = bucket_num
    # print("modified ex[0]:")
    # pp(examples[0])
    return examples, list_of_buckets


def categorize_test_set1(test_set, buckets):
    print("categorizing test set...")

    # print("len of test set:", len(test_set[0][0]))
    # pp(test_set)
    for col in range(1, len(test_set[0][0])):
        for i in range(len(test_set[0])):
            # print("i = {} / {}, col = {} / {}".format(i, len(test_set[0]), col, len(test_set[0][0])))
            for b in buckets[col - 1]:
                if b.start < test_set[0][i][col] <= b.end:
                    test_set[0][i][col] = b.label
                    # print("set bucket:", b.label)
                    break
            # print("bucket number:", bucket_num)
    # print("new test set:")
    # pp(test_set)
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
        # print("col:", col)
        column_freqs0 = {}
        column_freqs1 = {}
        for val in range(0, num_of_buckets):
            column_freqs0[val] = 1
            column_freqs1[val] = 1

        for ex, t in zip(examples[0], examples[1]):
            # print("ex, ", ex)
            # print("ex[{}] = {}".format(col, ex[col]))
            if t == 0:
                column_freqs0[ex[col]] += 1
            else:
                column_freqs1[ex[col]] += 1
        freqs0.append(column_freqs0)
        freqs1.append(column_freqs1)
    val_freqs.append(freqs0)
    val_freqs.append(freqs1)
    return val_freqs


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
    ex_cnt = 0
    errors = 0
    cnt0 = 0
    cnt1 = 0
    real0 = 0
    real1 = 0

    print("classify this examples: ")
    # pp(test_set)
    classes = []

    apriori_pr = find_class_freqs(train_set)
    train_len = len(train_set[0])
    for cls in apriori_pr.keys():
        apriori_pr[cls] /= train_len

    # cnt_this_cl_all = []
    # cnt_this_cl_all.append(get_elem(train_set, 0))
    # cnt_this_cl_all.append(get_elem(train_set, 1))
    # print(cnt_this_cl_all)
    class_freqs = get_freqs(train_set)
    # print("fr:", class_freqs)

    val_freqs = get_value_freqs(train_set)

    for ex, t in zip(test_set[0], test_set[1]):
        # print("classify:", ex)
        max_ans = float('-inf')
        my_class = float('-inf')

        for cl in {0, 1}:
            sum = 0
            # pp(train_set)
            # cnt_this_cl = get_elem(train_set, cl)
            cnt_this_cl = class_freqs[cl]
            # print("in class " + str(cl) + " " + str(cnt_this_cl) + " elems")

            for col in range(1, len(test_set[0][0])):
                cur_val = ex[col]
                # cnt_this_val_cl = get_elem(train_set, cl, col, cur_val)
                # print("col: " + str(col) + ", val: " + str(cur_val) + ", class: " + str(cl))
                # print(cnt_this_val_cl)
                # print(" ", val_freqs[cl][col][cur_val])

                this_pr = val_freqs[cl][col][cur_val] / cnt_this_cl

                # this_pr = cnt_this_val_cl / cnt_this_cl
                # print("this prob: " + str(this_pr) + ", log2: " + str(log2(this_pr)))
                sum += log2(this_pr)
            sum += log2(apriori_pr[cl])
            # print("sum = " + str(sum) + ", log2(pr(cl)): " + str(log2(apriori_pr[cl])))
            if sum >= max_ans:
                max_ans = sum
                my_class = cl
        classes.append(my_class)
        # print("for " + str(ex[0]) + " class is " + str(my_class) + ", real: " + str(t))
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

        ex_cnt += 1
        if ex_cnt % 100 == 0:
            print("cnt:", ex_cnt)
    print("errors: {} / {} => {}%, 0: {} vs {}, 1: {} vs {}".format(errors, len(classes), errors / len(classes) * 100, cnt0, real0, cnt1, real1))

    # for ex, cl, t in zip(test_set[0], classes, test_set[1]):
    #     # print("for " + ex[0] + " predict: " + str(cl) + ", real: " + str(t))
    #     if cl != t:
    #         errors += 1
    #     if cl == 0:
    #         cnt0 += 1
    #     else:
    #         cnt1 += 1
    #
    #     if t == 0:
    #         real0 += 1
    #     else:
    #         real1 += 1
    # print("errors: {} / {} => {}%, 0: {} vs {}, 1: {} vs {}".format(errors, len(classes), errors / len(classes) * 100, cnt0, real0, cnt1, real1))


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
        # print(ex)
        facts = list(str(ex[0])[8:].split())
        reqid = str(ex[1])[6:]
        query = str(ex[2])[6:]
        if query == "":
            continue

        if len(facts) != 1097:
            continue

        clicked = str(ex[3])[8:]

        cur_list = [query]
        cur_list += list(map(float, facts[:]))
        # if cnt0 == half_size and cnt1 < half_size and clicked == 'false':
        #     continue
        # if cnt1 == half_size and cnt0 < half_size and clicked == 'true':
        #     continue
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

        # if cnt0 >= half_size and cnt1 >= half_size:
        #     print("data is loaded")
        #     break

    # print("data:")
    # for d in enumerate(data):
    #     print(d)

    # print(data)
    # print('target:')
    # print(target)
    print("data is loaded")

    return data, target


def main():
    data, target = load_pool()
    # print("data:")
    # print(data)
    # print('target:')
    # print(target)
    split_ratio = 0.75
    training_set, test_set = generate_sets(data, target, split_ratio)
    modified_train_set, buckets = categorize_train_set1(training_set, num_of_buckets)

    # print("modified")
    # for i, j, c in zip(training_set[0], modified_train_set[0], training_set[1]):
    #     print(i, c)
    #     print(j, c)
    #     print()

    modified_test_set = categorize_test_set1(test_set, buckets)

    print("modified")
    # for i, j in zip(test_set[0], modified_test_set[0]):
    #     print(i)
    #     print(j)
    #     print()

    # for b in buckets:
    #     print(b)
    # print()
    # print()
    start = time.time()
    classify(modified_train_set, modified_test_set)
    end = time.time()
    print("time =", end - start)


main()
