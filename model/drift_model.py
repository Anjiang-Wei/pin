import numpy as np
import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

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

good = 0
total = 0
def compute_std(dic, keys):
    global good, total
    res = []
    for k in keys:
        res.append(dic[k])
    assert len(res) >= 1, res
    try:
        test = stats.normaltest(res)
        # print(test)
        if test.pvalue > 1e-3:
            good = good + 1
        # else:
        #     # sns.distplot(res, kde=True,bins=100)
        #     # plt.show()
        #     pass
        total += 1
    except Exception as e:
        pass
    return np.std(res)

def find_index(vals, low_val):
    for i in range(len(vals)):
        if vals[i] >= low_val:
            return i
    assert False, "Not Found, low_val is too high!"

def compute_sigma(bins = 100):
    for t in times[1:]:
        if t not in s1.keys():
            s1[t] = {}
        all_g = sorted(list(d1[t].keys()))
        interval = (max(all_g) - min(all_g)) / bins
        for idx in range(0, bins):
            low_idx = find_index(all_g, min(all_g) + idx * interval)
            high_idx = find_index(all_g, min(all_g) + (idx + 1) * interval)
            high_idx = max(low_idx + 1, high_idx)
            std = compute_std(d1[t], all_g[low_idx:high_idx])
            s1[t][idx] = std

            if idx not in s2.keys():
                s2[idx] = {}
            s2[idx][t] = std
        print(good, total, good / total)

def figure_2d_g():
    y, z = [], []
    for yy in s2.keys():
        y.append(yy)
        z.append(s1[0.1][yy])
    y, z = np.array(y), np.array(z)
    plt.plot(y, z)
    plt.show()

def compute_t():
    for t in s1.keys():
        vals = list(s1[t].values())
        print(t, sum(vals) / len(vals))


def figure_3d():
    fig = plt.figure()
    ax = Axes3D(fig)
    x, y, z = [], [], []
    for xx in s1.keys():
        for yy in s2.keys():
            x.append(xx)
            y.append(yy)
            z.append(s1[xx][yy])
    x, y, z = np.array(x), np.array(y), np.array(z)
    ax.plot_trisurf(x, y, z)
    plt.show()



if __name__ == "__main__":
    init()
    compute_d()
    compute_sigma()
    figure_2d_g()
    compute_t()
    # figure_3d()
