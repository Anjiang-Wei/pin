import numpy as np
import pprint
from scipy.stats.mstats_basic import trimmed_stde
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
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

s1 = {} # s1[timept][bin_idx] -> sigma
s2 = {} # s2[bix_idx][timept] -> sigma

# All the conductance (g) are converted to resistance if turning useR on
useR = True
model_char = "B"
config = {
    "A": [30 * 1e3, 3.227 * 1e3], # maxi, mini of resistance range
    "B": [70 * 1e3, 8 * 1e3]
}

def init():
    print("For model:", model_char)
    global times
    with open("../data/drift/model" + model_char + ".tsv", "r") as fin:
        lines = fin.readlines()
        for i in range(1, len(lines)):
            addr, timept, conductance, r = lines[i].split()
            addr, timept, conductance = int(addr), float(timept), float(conductance)
            if useR:
                conductance = 1 / conductance
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
    '''
    Compute the difference w.r.t time, given the same address [same initial g0]
    '''
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
last_normal_value = 0.0
def compute_std(dic, keys):
    global last_normal_value
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
            # sns.distplot(res, kde=True,bins=100)
            # plt.show()
            # pass
        total += 1
    except:
        return last_normal_value
    last_normal_value = np.std(res)
    return np.std(res)

def find_index(vals, low_val):
    for i in range(len(vals)):
        if vals[i] >= low_val:
            return i
    assert False, "Not Found, low_val is too high!"

def export(vals, append=True):
    vals_ = list(map(str, vals))
    with open("conf" + model_char, "a" if append else "w") as fout:
        fout.write(",".join(vals_) + "\n")

def compute_sigma(bins = 30):
    '''
    Given the number of bins (to segment [min_g, max_g]), compute the sigma for each bin
    Output: s1, s2
    '''
    global total, good
    # all_g_t = sorted(list(d1[times[-1]].keys()))
    max_val, min_val = config[model_char] # max(all_g_t), min(all_g_t)
    interval = (max_val - min_val) / bins
    export([max_val, min_val, bins], False)
    for t in times[1:]:
        if t not in s1.keys():
            s1[t] = {}
        all_g = sorted(list(d1[t].keys()))
        for idx in range(0, bins):
            low_idx = find_index(all_g, min_val + idx * interval)
            high_idx = find_index(all_g, min_val + (idx + 1) * interval)
            high_idx = max(low_idx + 1, high_idx)
            # if t == 0.01:
                # print(idx, high_idx - low_idx)
            std = compute_std(d1[t], all_g[low_idx:high_idx]) / (min_val + (idx+0.5) * interval)
            s1[t][idx] = std

            # export([t, idx, std])

            if idx not in s2.keys():
                s2[idx] = {}
            s2[idx][t] = std
        # print("good normal: ", good, total, good / total)
        good = total = 0
    

def figure_d_g():
    y, z = [], []
    print(min(d2.keys()), max(d2.keys()))
    for yy in list(filter(lambda x: x < config[model_char][0], d1[0.1].keys())):
        y.append(yy)
        z.append(d1[0.1][yy])
    y_, z_ = np.array(y), np.array(z)
    plt.plot(y_, z_, 'ro')
    plt.show()

def figure_s_g():
    y, z = [], []
    print(min(s2.keys()), max(s2.keys()))
    for tt in s1.keys():
        print(tt)
        for yy in s1[tt].keys():
            y.append(yy)
            z.append(s1[tt][yy])
            y_, z_ = np.array(y), np.array(z)
        plt.plot(y_, z_)
        plt.show()
        break

def figure_diff_t():
    # s1[timept][bin_idx] -> sigma
    # s2[bix_idx][timept] -> sigma
    print(times)
    x = []
    y = []
    cnt = 0
    for t in times[1:]:
        vals = list(d1[t].values())
        diff_considered = list(filter(lambda x: x < config[model_char][0], vals))
        diff_considered = list(map(lambda x: abs(x), diff_considered))
        mean_val = sum(diff_considered) / len(diff_considered)
        y.append(mean_val)
        x.append(math.log(t, 10))
        cnt += 1
    plt.plot(x, y)
    plt.show()
        
    
    #     print(t, sum(vals) / len(vals))
    #     y.append(sum(vals) / len(vals))
    #     x.append(cnt)
    #     cnt += 1
    # plt.plot(x, y)
    # plt.show()



def figure_sigma_t():
    # s1[timept][bin_idx] -> sigma
    # s2[bix_idx][timept] -> sigma
    print(times)
    x = []
    y = []
    cnt = 0
    for t in times[1:]:
        vals = list(s1[t].values())
        print(len(vals))
        print(t, sum(vals) / len(vals))
        y.append(sum(vals) / len(vals))
        x.append(cnt)
        cnt += 1
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    init()
    compute_d()
    figure_diff_t()
    compute_sigma()
    figure_sigma_t()
