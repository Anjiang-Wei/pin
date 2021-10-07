import pprint
import numpy as np
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
            if int(addr) not in addr_dict.keys():
                addr_dict[int(addr)] = [item]
            else:
                addr_dict[int(addr)].append(item)
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

def paint(vals, bins):
    sns.distplot(vals, kde=True,bins=bins)
    # plt.show()

def show_original_distr(time_stamp, time_dict):
    for ts in time_stamp:
        paint(extract_list(time_dict, 1, ts), 1000)
        plt.savefig("drift_figs/original_distr/" + ts + ".pdf")

def show_small_range(time_stamp, time_dict):
    def filter_res(vals):
        res = []
        min_, max_ = 1e-6, 1e-5
        for v in vals:
            if min_ <= v and v <= max_:
                res.append(v)
        return res

    for ts in time_stamp:
        res = extract_list(time_dict, 1, ts)
        res2 = filter_res(res)
        paint(res2, 1000)
        plt.savefig("drift_figs/small_range/" + ts + ".pdf")
    
def show_diff_abs(addr_dict):
    def compute_diff(val8, diffs):
        new_val8 = sorted(val8, key=lambda x: x[0])
        time0 = new_val8[0]
        g0 = time0[1]
        for i in range(1, 8):
            real_diff = new_val8[i][1] - g0
            diffs[i] = diffs.get(i, []) + [real_diff]
    diffs = {}
    for addr in range(80000, 96383+1):
        if len(addr_dict[addr]) == 8:
            compute_diff(addr_dict[addr], diffs)
    
    for i in range(1, 8):
        paint(diffs[i], 1000)
        plt.savefig("drift_figs/diff_time/" + str(i) + ".pdf")

def show_diff_g0(addr_dict):
    min_, max_ = 0.0, 0.00035
    interval = (max_ - min_) / 10
    
    def determine_bin(x, minv, maxv, interval):
        for i in range(0, 10):
            i_start = minv + i * interval
            i_end = minv + (i + 1) * interval
            if i_start <= x and x <= i_end:
                return i
        assert False, x
    
    def compute_diff(val8, diffs, time_stamp_idx=3):
        new_val8 = sorted(val8, key=lambda x: x[0])
        time0 = new_val8[0]
        g0 = time0[1]
        real_diff = new_val8[time_stamp_idx][1] - g0
        bin_idx = determine_bin(g0, min_, max_, interval)
        diffs[bin_idx] = diffs.get(bin_idx, []) + [real_diff]
    
    diffs = {}
    for addr in range(80000, 96383+1):
        if len(addr_dict[addr]) == 8:
            compute_diff(addr_dict[addr], diffs)
    
    for i in range(8, 9):# (0, 7), (7, 8), (8, 9)
        paint(diffs[i], 1000)
        plt.savefig("drift_figs/diff_g0/" + str(i) + ".pdf")
    

if __name__ == "__main__":
    init()
    res = extract_list(time_dict, 1)
    time_stamp = ["0.0", "0.01", "0.1", "1.0", "100.0", "1000.0", "10000.0", "100000.0"]
    # strange(time_stamp, time_dict)
    # show_original_distr(time_stamp, time_dict)
    # show_small_range(time_stamp, time_dict)
    # show_diff_abs(addr_dict)
    show_diff_g0(addr_dict)
