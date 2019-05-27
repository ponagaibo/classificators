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
    elif file == "shuffled":
        filename = "\shuf"
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
        len_c = len(clicks)
        if len_c == 0:
            continue
        # if len_c != 10 and len_c != 15:
        #     print("{}: len of clicks: {}".format(cnt, len(clicks)))

        if len(facts) != len_of_facts:
            continue

        cur_list = [query]
        query_facts = facts[:]  #

        if my_features:
            # print("clicks:", clicks)
            num_of_clicks = sum(clicks)
            clicks1 = clicks[0]
            clicks2 = sum(clicks[:2])
            clicks3 = sum(clicks[:3])
            clicks4 = sum(clicks[:4])
            clicks5 = sum(clicks[:5])
            clicks6 = sum(clicks[:6])
            clicks7 = sum(clicks[:7])
            clicks8 = sum(clicks[:8])
            clicks9 = sum(clicks[:9])
            clicks10 = sum(clicks[:10])
            clicks11 = sum(clicks[:11])
            clicks12 = sum(clicks[:12])
            clicks13 = sum(clicks[:13])
            clicks14 = sum(clicks[:14])
            clicks15 = sum(clicks[:15])
            num_clicked_sport = 0
            num_clicked_fh = 0

            # print("sum:", num_of_clicks)
            sport_hosts = open(r'C:\Users\Anastasiya\Desktop\диплом\project\classificators\parser\orig_sport_hosts.txt',
                           'r', encoding='utf-8')
            sport_hosts_list = sport_hosts.read().split()
            sport_hosts.close()

            f_hosts = open(r'C:\Users\Anastasiya\Desktop\диплом\project\classificators\parser\football_hosts.txt',
                           'r', encoding='utf-8')
            f_h_hosts_list = f_hosts.read().split()
            f_hosts.close()

            hosts_to_check = {}
            cnt_ = 0
            top1 = False
            top2 = False
            top3 = False
            top4 = False
            top5 = False
            top6 = False
            top7 = False
            top8 = False
            top9 = False
            top10 = False
            top11 = False
            top12 = False
            top13 = False
            top14 = False
            top15 = False

            top1_fh = False
            top2_fh = False
            top3_fh = False
            top4_fh = False
            top5_fh = False
            top6_fh = False
            top7_fh = False
            top8_fh = False
            top9_fh = False
            top10_fh = False
            top11_fh = False
            top12_fh = False
            top13_fh = False
            top14_fh = False
            top15_fh = False
            sport_amount = 0
            fh_amount = 0
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

                best_host_sport = ""
                best_host_fh = ""
                place_of_best_host_sport = 0
                place_of_best_host_fh = 0
                if host_with_two_dirs != "":
                    if hosts_to_check.get(host_with_two_dirs) is not None:
                        continue
                    # print("!   " + host_with_two_dirs + " in sport! place: " + str(cnt_))
                    hosts_to_check[host_with_two_dirs] = cnt_
                    if host_with_two_dirs in sport_hosts_list:
                        best_host_sport = host_with_two_dirs
                        place_of_best_host_sport = cnt_
                    if host_with_two_dirs in f_h_hosts_list:
                        best_host_fh = host_with_two_dirs
                        place_of_best_host_fh = cnt_

                if host_with_one_dir != "":
                    if hosts_to_check.get(host_with_one_dir) is not None:
                        continue
                    # print("!   " + host_with_one_dir + " in sport! place: " + str(cnt_))
                    hosts_to_check[host_with_one_dir] = cnt_
                    if host_with_one_dir in sport_hosts_list:
                        best_host_sport = host_with_one_dir
                        place_of_best_host_sport = cnt_
                    if host_with_one_dir in f_h_hosts_list:
                        best_host_fh = host_with_one_dir
                        place_of_best_host_fh = cnt_

                if hosts_to_check.get(host) is not None:
                    continue
                hosts_to_check[host] = cnt_
                if host in sport_hosts_list:
                    best_host_sport = host
                    place_of_best_host_sport = cnt_
                if host in f_h_hosts_list:
                    best_host_fh = host
                    place_of_best_host_fh = cnt_

                if best_host_sport != "":
                    if clicks[place_of_best_host_sport-1] == 1:
                        num_clicked_sport += 1
                    sport_amount += 1
                    # print("!   " + best_host_sport + " in sport! place: " + str(cnt_))
                    if cnt_ == 1:
                        top1 = True
                    if cnt_ <= 2:
                        top2 = True
                    if cnt_ <= 3:
                        top3 = True
                    if cnt_ <= 4:
                        top4 = True
                    if cnt_ <= 5:
                        top5 = True
                    if cnt_ <= 6:
                        top6 = True
                    if cnt_ <= 7:
                        top7 = True
                    if cnt_ <= 8:
                        top8 = True
                    if cnt_ <= 9:
                        top9 = True
                    if cnt_ <= 10:
                        top10 = True
                    if cnt_ <= 11:
                        top11 = True
                    if cnt_ <= 12:
                        top12 = True
                    if cnt_ <= 13:
                        top13 = True
                    if cnt_ <= 14:
                        top14 = True
                    if cnt_ <= 15:
                        top15 = True

                if best_host_fh != "":
                    if clicks[place_of_best_host_fh-1] == 1:
                        num_clicked_fh += 1
                    fh_amount += 1
                    # print("!   " + best_host_sport + " in sport! place: " + str(cnt_))
                    if cnt_ == 1:
                        top1_fh = True
                    if cnt_ <= 2:
                        top2_fh = True
                    if cnt_ <= 3:
                        top3_fh = True
                    if cnt_ <= 4:
                        top4_fh = True
                    if cnt_ <= 5:
                        top5_fh = True
                    if cnt_ <= 6:
                        top6_fh = True
                    if cnt_ <= 7:
                        top7_fh = True
                    if cnt_ <= 8:
                        top8_fh = True
                    if cnt_ <= 9:
                        top9_fh = True
                    if cnt_ <= 10:
                        top10_fh = True
                    if cnt_ <= 11:
                        top11_fh = True
                    if cnt_ <= 12:
                        top12_fh = True
                    if cnt_ <= 13:
                        top13_fh = True
                    if cnt_ <= 14:
                        top14_fh = True
                    if cnt_ <= 15:
                        top15_fh = True
            # print("top1: {}, top3: {}, top5: {}, top7: {}, top10: {}, top15: {}".
            #       format(str(top1), str(top3), str(top5), str(top7), str(top10), str(top15)))

            # query_facts.append(int(top1))
            # query_facts.append(int(top2))
            # query_facts.append(int(top3))
            # query_facts.append(int(top4))
            # query_facts.append(int(top5))
            # query_facts.append(int(top6))
            # query_facts.append(int(top7))
            # query_facts.append(int(top8))
            # query_facts.append(int(top9))
            query_facts.append(int(top10))
            query_facts.append(int(top10_fh))
            # query_facts.append(int(top11))
            # query_facts.append(int(top12))
            # query_facts.append(int(top13))
            # query_facts.append(int(top14))
            # query_facts.append(int(top15))

            query_facts.append(num_of_clicks)

            # query_facts.append(clicks1)
            # query_facts.append(clicks2)
            # query_facts.append(clicks3)
            # query_facts.append(clicks4)
            # query_facts.append(clicks5)
            # query_facts.append(clicks6)
            # query_facts.append(clicks7)
            # query_facts.append(clicks8)
            # query_facts.append(clicks9)
            query_facts.append(clicks10)  # на 3 месте
            # query_facts.append(clicks11)
            # query_facts.append(clicks12)
            # query_facts.append(clicks13)
            # query_facts.append(clicks14)
            # query_facts.append(clicks15)

            query_facts.append(sport_amount)
            query_facts.append(fh_amount)
            query_facts.append(num_clicked_sport)
            query_facts.append(num_clicked_fh)

        # print("new query facts:", query_facts)
        cur_list += list(map(float, query_facts))

        if half_size is not None:
            if cnt0 == half_size and cnt1 < half_size and targ == 0:
                continue
            if cnt1 == half_size and cnt0 < half_size and targ == 1:
                continue

        data.append(cur_list)
        if targ == 0:
            target.append(0)  # not inverted!!!!
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
