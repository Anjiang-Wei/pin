import pprint
def extract(l, prefix):
    assert prefix in l
    return l.replace(prefix, "")[:5]

m = {}
with open("error_log", "r") as fin:
    lines = fin.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("../data/"):
            write_error = extract(lines[i+1], "write_range_error_rate: ")
            read_error = extract(lines[i+2], "read_range_error_rate: ")
            m[line.replace("../data/", "")] = [write_error, read_error]
            i += 3
        else:
            i += 1

# pprint.pprint(m)

def filter_map(keywords):
    res = {}
    for k in m.keys():
        match = True
        for word in keywords:
            if word not in k:
                match = False
        if match:
            res[k] = m[k]
    return res

method = ["fppv", "ispp", "sdr"]
bpc = ["2bpc", "3bpc"]
chip = ["chip1", "chip2"]
for b in bpc:
    for c in chip:
        for me in method:
            print(me, c, b)
            result = filter_map([me, b, c])
            pprint.pprint(result)

# result = filter_map([method[1], chip[0]])
# pprint.pprint(result)
