import pprint
# header:
# addr	timept	g	range
addr_dict = {}
with open("../data/drift/modelA.tsv", "r") as fin:
    lines = fin.readlines()
    for i in range(1, len(lines)):
        addr, timept, g, r = lines[i].split()
        item = [float(timept), float(g), int(r)]
        if addr not in addr_dict.keys():
            addr_dict[addr] = [item]
        else:
            addr_dict[addr].append(item)

pprint.pprint(addr_dict)
