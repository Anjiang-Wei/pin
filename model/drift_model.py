import pprint
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

def extract_list(dic, index):
    '''
    Return the list containing the values indicated by index for the dictionary
    '''
    res = []
    for val in dic.values():
        for item in val:
            res.append(item[index])
    return res
# pprint.pprint(addr_dict.keys())
# pprint.pprint(time_dict.keys())
if __name__ == "__main__":
    init()
    res = extract_list(time_dict, 0)
    print(len(res) / 8)
