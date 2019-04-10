import pandas as pd


def load_pool():
    df = pd.read_csv(r'C:\Users\Anastasiya\Desktop\диплом\pool_tv_20190406', delimiter='\t', encoding='utf-8',
                     names=['factors', 'reqid', 'query', 'clicked'])
    headers = ['factors', 'reqid', 'query', 'clicked']
    pool_size = df.__len__()
    cnt = 0
    data = []
    target = []
    for ex in df.values:
        print(ex)
        facts = list(str(ex[0])[8:].split())
        reqid = str(ex[1])[6:]
        query = str(ex[2])[6:]
        clicked = str(ex[3])[8:]
        # print("f:", facts)
        # print("r:", reqid)
        # print("q:", query)
        # print("c:", clicked)
        cur_list = [query]
        a = [i for i in range(5)]
        cur_list += list(map(float, facts))
        data.append(cur_list)
        if clicked == 'false':
            target.append(0)
        else:
            target.append(1)
        # cur_tar = int(bool(clicked))
        # target.append(cur_tar)

        # print("len of facts:", len(facts))
        print()
        cnt += 1
        if cnt == 15:
            break

    print("data:")
    print(data)
    print('target:')
    print(target)

def main():
    load_pool()

main()