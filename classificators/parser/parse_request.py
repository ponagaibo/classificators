# -*- coding: utf8 -*-
from bs4 import BeautifulSoup
import requests
import pandas as pd
from pprint import pprint as pp
import time

headers = None


# TODO: check parsing

def load_tsv():
    df = pd.read_csv('little_data.tsv', delimiter='\t', encoding='utf-8')
    global headers
    headers = df.dtypes.index.values
    data_feature_names = headers[:-1]
    file_data = df[data_feature_names]
    target_name = headers[-1]
    file_target = df[target_name]
    return file_data, file_target


data, target = load_tsv()
# for d in data: # prints columns
#     pp(data[d])

pp(data['text'])

list_top3 = []
list_top5 = []
list_top10 = []

for t in data['text']:
    print()
    print(t)
    request_text = t
    words_of_request = request_text.split()

    start_url = "https://yandex.ru/search/?text="
    end_url = "&lr=213"
    key_pass = "/xml?user=ponagaibo&key=03.129922212:6edf0fa5507f344a609be991e364b7b2"
    delim = "%20"
    url = start_url + delim.join(words_of_request) + end_url + key_pass

    page = requests.get(url)
    page.encoding = 'utf-8'
    page_text = page.text
    soup = BeautifulSoup(page_text, features="html.parser")
    print(soup.prettify())

    f_pret = open('data.txt', 'w', encoding='utf-8')
    f_pret.write(soup.prettify())
    f_pret.close()

    f_hosts = open('football_hosts.txt', 'r', encoding='utf-8')
    sport_hosts = f_hosts.read().split()
    f_hosts.close()

    list_of_links = []
    hosts_to_check = {}
    cnt = 0
    for link in soup.find_all('a', href=True):
        if len(link['href']) <= 1:
            continue
        if link['href'][0:2] == "//":
            continue
            # pass "//yandex.ru/"
        if link['href'][0:7] == "/search":
            continue
        # print("\n    link:")
        # print(link['href'])

        whole_url = link['href'].split("/")
        whole_url = list(filter(None, whole_url))

        # print("    parsed link (whole_url):")
        # print(whole_url)

        host = whole_url[1]
        list_of_host_words = host.split(".")

        # print("    host (list_of_host_words):")
        # print(list_of_host_words)

        if list_of_host_words[1] == "yandex" or list_of_host_words[0] == 'yandex':
            # print("  ~~~yandex")
            continue

        if list_of_host_words[0] == "www":
            del(list_of_host_words[0])

        host = ".".join(list_of_host_words)

        host_with_dir = ""
        if len(whole_url) >= 3:
            directory = whole_url[2]
            # print("      dir: " + directory)
            host_with_dir = host + "/" + directory

        host_with_two_dirs = ""
        if len(whole_url) >= 4:
            # print("      second dir: " + whole_url[3])
            host_with_two_dirs = host_with_dir + "/" + whole_url[3]

        # print("        ***hosts: " + host + "..." + host_with_dir + "..." + host_with_two_dirs + ".")

        if (hosts_to_check.get(host_with_two_dirs) is None
            and hosts_to_check.get(host_with_dir) is None
            and len(whole_url) >= 3):
            cnt += 1

        if host_with_two_dirs != "":
            # print("    cur host_with_two_dirs: " + str(host_with_two_dirs))
            if hosts_to_check.get(host_with_two_dirs) is not None:
                # print("host_with_two_dirs already exist")
                continue
            hosts_to_check[host_with_two_dirs] = cnt

        if host_with_dir != "":
            # print("    cur host_with_dir: " + str(host_with_dir))
            if hosts_to_check.get(host_with_dir) is not None:
                # print("host_with_dir already exist")
                continue
            hosts_to_check[host_with_dir] = cnt

        if hosts_to_check.get(host) is not None:
            # print("host already exist")
            continue
        hosts_to_check[host] = cnt

        if cnt >= 11:
            break
        time.sleep(2)

    break

    top3 = False
    top5 = False
    top10 = False

    for i, j in hosts_to_check.items():
        print(str(i) + " : " + str(j))
        if i in sport_hosts:
            print("   " + str(i) + " in sport! place: " + str(j))
            if j <= 3:
                top3 = True
            if j <= 5:
                top5 = True
            if j <= 10:
                top10 = True
    print("top3: " + str(top3) + ", top5: " + str(top5) + ", top10: " + str(top10))
    list_top3.append(top3)
    list_top5.append(top5)
    list_top10.append(top10)

pp(list_top3)
