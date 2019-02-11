# -*- coding: utf8 -*-
from bs4 import BeautifulSoup
import requests

request_text = "футбольный матч"
words_of_request = request_text.split()

start_url = "https://yandex.ru/search/?text="
end_url = "&lr=213"
delim = "%20"
url = start_url + delim.join(words_of_request) + end_url

page = requests.get(url)
data = page.text
soup = BeautifulSoup(data, features="html.parser")

f_pret = open('data.txt', 'w', encoding='utf-8')
f_pret.write(soup.prettify())
f_pret.close()

f_links = open('links.txt', 'w', encoding='utf-8')
list_of_links = []
for link in soup.find_all('a', class_="link link_theme_outer path__item i-bem"):
    hasLink = link.b
    if hasLink != None:
        f_links.write(link.b.string)
        f_links.write("\n")
        list_of_links.append(link.b.string)
f_links.close()
print(list_of_links)

