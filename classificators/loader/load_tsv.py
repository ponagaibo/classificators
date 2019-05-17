import csv
import pandas as pd
from pprint import pprint as pp


def get_row():
    with open(r'C:\Users\Anastasiya\Desktop\диплом\sport_pool_20190305_20190307', 'r', encoding='utf-8', newline='\n') as csvfile:
        data_reader = csv.reader(csvfile)
        for row in data_reader:
            tmp_list = str(row[0]).split('\t')

            del(tmp_list[-1])
            tmp_list = list(map(str, tmp_list))
            if len(tmp_list) != 4:
                continue

            yield tmp_list


def load_pool_iter(half_size=None, stop_size=None):
    cnt = 0
    data = []
    target = []
    cnt0 = 0
    cnt1 = 0
    len_of_facts = 1113  #len(list(str(df.values[0][1])[8:].split()))
    print("len:", len_of_facts)
    for ex in get_row():
        query = str(ex[0])[6:]
        if query == "":
            continue

        facts = list(str(ex[1])[8:].split())
        urls = list(str(ex[2])[5:].split())
        targ = int(str(ex[3])[7:])

        if len(facts) != len_of_facts:
            continue

        cur_list = [query]
        cur_list += list(map(float, facts[:]))

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
    print("cnt0 = {}, cnt1 = {}".format(cnt0, cnt1))
    print("data is loaded")
    return data, target


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

    # print("len:", len_of_facts)
    for ex in df.values:
        query = str(ex[0])[6:]

        if query == "":
            continue

        facts = list(str(ex[1])[8:].split())
        # urls = list(str(ex[2])[5:].split())
        targ = int(str(ex[3])[7:])
        # clicks = list(str(ex[4])[7:].split())

        if len(facts) != len_of_facts:
            continue

        cur_list = [query]
        cur_list += list(map(float, facts[:]))

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
