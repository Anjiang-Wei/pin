import numpy as np
import pprint

# header:
# addr	timept	g (conductance)  range

# g1[timept][addr] -> conductance
# g2[addr][timept] -> conductance
g1 = {}
g2 = {}
times = [] # [0.0, 0.01, 0.1, 1.0, 100.0, 1000.0, 10000.0, 100000.0]
addrs = []

# d1[timept][origin_g0] -> g_t - g_0, s1 -> sigma
# d2[origin_g0][timept] -> g_t - g_0, s2 -> sigma
d1 = {}
d2 = {}

s1 = {}
s2 = {}

def init():
    global times
    with open("../data/drift/modelA.tsv", "r") as fin:
        lines = fin.readlines()
        for i in range(1, len(lines)):
            addr, timept, conductance, r = lines[i].split()
            addr, timept, conductance = int(addr), float(timept), float(conductance)
            if timept not in g1.keys():
                g1[timept] = {}
            g1[timept][addr] = conductance
            if addr not in g2.keys():
                g2[addr] = {}
            g2[addr][timept] = conductance
    times = sorted(list(g1.keys()))

    #Record those addr with 8 entries
    for addr in g2.keys():
        if len(g2[addr]) == 8:
            addrs.append(addr)

def compute_d():
    for t in times[1:]:
        if t not in d1.keys():
            d1[t] = {}
        for addr in g1[t].keys():
            g_0 = g1[0][addr]
            diff = g1[t][addr] - g_0
            d1[t][g_0] = diff

            if g_0 not in d2.keys():
                d2[g_0] = {}
            d2[g_0][t] = diff

def compute_sigma(num = 100):
    for t in times[1:]:
        if t not in s1.keys():
            s1[t] = {}
        all_g = sorted(list(d1[t].keys()))
        print(all_g)

if __name__ == "__main__":
    init()
    compute_d()
    compute_sigma()
