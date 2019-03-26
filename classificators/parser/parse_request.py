# -*- coding: utf8 -*-
from bs4 import BeautifulSoup
import requests
import pandas as pd
from pprint import pprint as pp

headers = None


# TODO: check parsing

def load_tsv():
    df = pd.read_csv('little_data.tsv', delimiter='\t', encoding='utf-8')
    global headers
    headers = df.dtypes.index.values
    data_feature_names = headers[:-1]

    file_data = df[data_feature_names]

    # pp(data)
    # sLength = len(df['has_sport'])
    # mm = [i*i for i in range(sLength)]
    # data['toppp']=mm
    # pp(data)

    target_name = headers[-1]
    file_target = df[target_name]

    # data = data.assign(class_sport=pd.Series(target).values)
    # pp(data)

    return file_data, file_target


data, target = load_tsv()
# for d in data: prints columns
#     pp(data[d])

# pp(data['text'][0])

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
    # print(url)

    page = requests.get(url)
    page.encoding = 'utf-8'
    page_text = page.text
    soup = BeautifulSoup(page_text, features="html.parser")
    print(soup.prettify())
    break

    f_pret = open('data.txt', 'w', encoding='utf-8')
    f_pret.write(soup.prettify())
    f_pret.close()

    # f_links = open('links.txt', 'w', encoding='utf-8')
    # f_test = open('test.txt', 'w', encoding='utf-8')

    f_hosts = open('sport_hosts.txt', 'r', encoding='utf-8')
    sport_hosts = f_hosts.read().split()
    f_hosts.close()

    list_of_links = []
    hosts_to_check = {}
    cnt = 1

    for link in soup.find_all('a', href=True):
        # print("    link:")
        # print(link['href'])
        if link['href'][0:2] == "//":
            continue

        whole_url = link['href'].split("/")
        whole_url = list(filter(None, whole_url))

        if len(whole_url) <= 2:
            # print("  ~~~without dir")
            continue

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
        directory = whole_url[2]
        # print("      dir: " + directory)
        host_with_dir = host + "/" + directory
        # print("        ***hosts: " + host + "    " + host_with_dir)

        if hosts_to_check.get(host) is not None:
            continue
        if hosts_to_check.get(host_with_dir) is not None:
            continue
        hosts_to_check[host] = cnt
        hosts_to_check[host_with_dir] = cnt
        cnt += 1
        sleep(1)


    # f_links.close()
    # f_test.close()

    top3 = False
    top5 = False
    top10 = False

    for i, j in hosts_to_check.items():
        print(i)
        if i in sport_hosts:
            # print("   " + str(i) + " in sport! place: " + str(j))
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
