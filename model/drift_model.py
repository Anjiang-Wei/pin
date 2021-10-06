import pprint
import seaborn as sns
import matplotlib.pyplot as plt

# header:
# addr	timept	g	range
addr_dict = {} # addr -> list of [timept, g, range]
time_dict = {} # timept -> list of [addr, g, range]
def init():
    with open("../data/drift/modelA.tsv", "r") as fin:
        lines = fin.readlines()
        for i in range(1, len(lines)):
            addr, timept, g, r = lines[i].split()
            item = [float(timept), float(g), int(r)]
            if addr not in addr_dict.keys():
                addr_dict[addr] = [item]
            else:
                addr_dict[addr].append(item)
            item2 = [int(addr), float(g), int(r)]
            if timept not in time_dict.keys():
                time_dict[timept] = [item2]
            else:
                time_dict[timept].append(item2)

def extract_list(dic, index, dic_key=None):
    '''
    Return the list containing the values indicated by index for the dictionary
    '''
    res = []
    if dic_key == None:
        for val in dic.values():
            for item in val:
                res.append(item[index])
        return res
    else:
        for item in dic[dic_key]:
            res.append(item[index])
        return res

def strange(time_stamp, time_dict):
    t = []
    for ts in time_stamp:
        t.append(len(extract_list(time_dict, 1, ts)))
    print("Inequal measurement:", t)

if __name__ == "__main__":
    init()
    res = extract_list(time_dict, 1)
    time_stamp = ["0.0", "0.01", "0.1", "1.0", "100.0", "1000.0", "10000.0", "100000.0"]
    strange(time_stamp, time_dict)
    