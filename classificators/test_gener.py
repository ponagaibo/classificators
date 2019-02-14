# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict

d = defaultdict(list)
dB = defaultdict(list)
totalcnt = 0

n, m = input().split()
for i in range(int(n)):
    word = input()
    totalcnt += 1
    d[word].append(i + 1)

for i in range(int(m)):
    word = input()
    totalcnt += 1
    if word in d:
        print(" ".join(map(str, (d[word]))))
    else:
        print(-1)
    # dB[word]
    # d[word]


print(totalcnt)


'''
for k, v in dB.items():
    print(k, end="    ")
    if d[k] != []:
        for i in d[k]:
            print(str(i), end = " ")
        print()
    else:
        print(str(-1))

print()
print(d['z'])
print()

for i in d.items():
    print(i)
'''