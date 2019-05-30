import sys

from math import ceil
import random
from pprint import pprint as pp
import math
import time
from loader import load_tsv
import cProfile


MAX_DEPTH = 10  # 40
NUM_OF_BUCKETS = 8  # 16
LEAF_SIZE = 431  # 55
UNIFORM = False
# попробовать 256 33, 400 301, 415 11, 128 11, 256 101, 400 33


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

    # инициализация:
    for i in range(NUM_OF_BUCKETS):  # для каждого возможного значения фичи (бакета)
        tmp_features = []
        for j in range(n_features + 1):  # для каждой фичи создаем массив под классы
            tmp_classes = []
            for k in range(2):  # в него добавляем нули для каждого класса
                tmp_classes.append(0)
            tmp_features.append(tmp_classes)  # добавляем такой массив каждой фиче
        all_freqs.append(tmp_features)

    # подсчет:
    for ex, t in zip(examples[0], examples[1]):  # смотрим текущий пример и класс
        for col in range(1, n_features + 1):  # смотрим все фичи
            all_freqs[ex[col]][col][t] += 1  # [bucket] [feature_num] [class] => NUM_OF_BUCKETS x 1113 x 2
            # ex[col] => получаем значение фичи под номером col
            # увеличиваем счетчик для этого бакета, для фичи с номером=col и классом=t
    return all_freqs


def best_question(examples):
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


def build_tree(examples, cur_depth):
    if len(examples[0]) < LEAF_SIZE:
        # print("    Max leaf size is reached!")
        return Leaf(examples)
    if cur_depth == MAX_DEPTH:
        # print("    Max depth is reached!")
        return Leaf(examples)

    gain, question = best_question(examples)
    if gain == 0:
        return Leaf(examples)

    true_branch, false_branch = divide_data(examples, question)
    # if len(true_branch[0]) < LEAF_SIZE or len(false_branch[0]) < LEAF_SIZE:
    #     # print("Max leaf size is reached!")
    #     return Leaf(examples)

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
        if target != main_class(node.predictions, 1): # changed!!
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

        print("cur err: {}, pes err: {}".format(cur_err, pes_err))
        if cur_err > pes_err:
            print("    !!! Do pruning !!!")
            return Leaf(None, tree.true_branch.predictions, tree.false_branch.predictions, leafs_err)
        else:
            return tree
    return tree


def measure(dataset, tree, weight, train_size, pivot):
    train_classes = []
    train_data = dataset[0]
    train_target = dataset[1]
    for i in range(len(dataset[0])):
        c = classify(train_data[i], tree)
        # print("c:", c)
        train_classes.append(c)

    errors = 0
    tp = 1
    tn = 1
    fn = 1
    fp = 1
    rand_errors = 0
    for i in range(len(dataset[0])):
        my_class = main_class(train_classes[i], weight)
        rand_class = random.randint(0, train_size - 1)
        t = train_target[i]
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
    print("tp: {}, fn: {}, fp: {}, tn: {}".format(tp - 1, fn - 1, fp - 1, tn - 1))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_measure = 2 * recall * precision / (recall + precision)

    print("    errors: {} / {} => {}%,  F1: {},  R: {}, P: {}"
          .format(errors, len(train_target), round(errors / len(train_target) * 100, 3),
                  round(f1_measure, 3), round(recall, 3), round(precision, 3)))
    print("    random error: {} / {} => {}%".format(rand_errors, len(train_target),
                                                    round(rand_errors / len(train_target) * 100, 3)))
    return f1_measure


def main():
    my_f = False
    filename = "shuffled"
    data, target = load_tsv.load_pool(filename, my_features=my_f, stop_size=50000)

    print("c4.5")
    print("file:", filename)
    print("my features:", my_f)
    print("MAX_DEPTH =", MAX_DEPTH)
    print("NUM_OF_BUCKETS =", NUM_OF_BUCKETS)
    print("LEAF_SIZE =", LEAF_SIZE)

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

    print("building tree...")
    root = build_tree(modified_train_set, 0)
    # print_tree(root)

    # for i in range(len(modified_test_set[0])):
    #     count_errors(train_data[i], train_target[i], root)
    # print("pruning...")
    # count_and_prune(root, modified_train_set)

    # print("    adding weights to values in class...")
    weight = class_fract  # при 0.22 ставит мало класса 1, при 0.6 чаще
    print("    weight =", weight)
    # print("checking training set...")
    # measure(modified_train_set, root, weight, train_size, pivot)

    print("classifying...")
    f1_m = measure(modified_test_set, root, weight, train_size, pivot)
    return f1_m


start_time = time.time()

# pr = cProfile.Profile()
# pr.enable()

orig_stdout = sys.stdout
f = open(r'C:\Users\Anastasiya\Desktop\диплом\outs\dt_50k_16b_twice_without_w', 'w')
sys.stdout = f
NUM_OF_BUCKETS = 16
for MAX_DEPTH in range(2, 22):
    main()
for MAX_DEPTH in range(2, 22):
    main()
sys.stdout = orig_stdout
f.close()

# pr.disable()

# main()

end_time = time.time()
diff = end_time - start_time
print("~~~ %s sec ~~~" % diff)
print("~ {} min ~".format(diff / 60))

# pr.print_stats(sort="cumtime")
