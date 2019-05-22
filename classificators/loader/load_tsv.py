import csv
import pandas as pd
from pprint import pprint as pp


def load_pool(file, half_size=None, stop_size=None, my_features=False):
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

        if len(facts) != len_of_facts:
            continue

        cur_list = [query]
        query_facts = facts[:]  #

        if my_features:
            # print("clicks:", clicks)
            num_of_clicks = sum(clicks)
            # print("sum:", num_of_clicks)
            f_hosts = open(r'C:\Users\Anastasiya\Desktop\диплом\project\classificators\parser\football_hosts.txt', 'r', encoding='utf-8')
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
            # query_facts.append(int(top1))
            # query_facts.append(int(top3))
            # query_facts.append(int(top5))
            # query_facts.append(int(top7))
            # query_facts.append(int(top10))
            # query_facts.append(int(top15))
            query_facts.append(num_of_clicks)

        # print("new query facts:", query_facts)
        cur_list += list(map(float, query_facts))

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
