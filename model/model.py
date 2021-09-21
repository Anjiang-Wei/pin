import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# names = ['addr', 'nreads', 'nsets', 'nresets', 'rf', 'if', 'rlo', 'rhi', 'success', 'attempts1', 'attempts2']

# 0 addr
# 4 rf (read final)
# 6 rlo (range low)
# 7 rhi (range high)
# 8 success

filename = ""
data = {}
length = 0
error_rate = 0.0
low_high = {}

def data_init():
    def add_to_dict(dct, entry_name, init_value, append_item):
        if entry_name not in dct.keys():
            dct[entry_name] = init_value
        dct[entry_name].append(append_item)
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for i in range(0, len(lines)):
            splitted = lines[i].split()
            add_to_dict(data, "addr", [], int(float(splitted[0])))
            add_to_dict(data, "rf", [], float(splitted[4]))
            add_to_dict(data, "rlo", [], float(splitted[6]))
            add_to_dict(data, "rhi", [], float(splitted[7]))
            add_to_dict(data, "success", [], int(float(splitted[8])))

def data_validate():
    def extract_features():
        global length, error_rate, low_high
        length = len(data["addr"])
        low_high = list(dict(zip(data["rlo"], data["rhi"])).items())
        error_rate = data["success"].count(0) / length
    
    def print_features():
        print("length", length)
        print("error_rate", error_rate)
        print("low_high", low_high)
    
    def validate_write_success(low, high, final, success):
        if final >= low and final <= high:
            assert success == 1
        else:
            assert success == 0
    
    extract_features()
    print_features()
    for i in range(length):
        validate_write_success(data["rlo"][i], data["rhi"][i], data["rf"][i], data["success"][i])

def draw():
    '''
    Originate from exptdata\retention\dist-conductance.py
    '''
    nranges = len(set(data["rlo"]))
    assert nranges == len(low_high)
    lows, highs = sorted(list(set(data["rlo"]))), sorted(list(set(data["rhi"])))
    if nranges == 8:
        Manual_Rmins = np.array([0.0001, 4.38, 4.84, 5.42, 6.16, 7.19, 9.23, 35])
        Manual_Rmaxs = np.array([4.3, 4.75, 5.3, 6.01, 6.99, 8.9, 25, 10000])
    elif nranges == 4:
        Manual_Rmins = np.array([0.0001, 5.38, 6.93, 18])
        Manual_Rmaxs = np.array([5.1, 6.48, 14, 10000])
    Write_Rmins = np.array(lows) / 1000
    Write_Rmaxs = np.array(highs) / 1000
    Manual_Gmaxs, Manual_Gmins = 1 / Manual_Rmins / 1000, 1 / Manual_Rmaxs / 1000
    Write_Gmaxs, Write_Gmins = 1 / Write_Rmins / 1000, 1 / Write_Rmaxs / 1000


    # Load data and process
    pd_data = pd.DataFrame.from_dict(data)
    def compute_bin(x):
        for i in range(0, len(Manual_Rmins) - 1):
            if Manual_Rmins[i] <= x and x < Manual_Rmins[i + 1]:
                return i
        return len(Manual_Rmins) - 1

    pd_data['g'] = 1 / pd_data['rf']
    pd_data['rf'] = pd_data['rf'] / 1000
    pd_data['bin'] = pd_data["rf"].map(compute_bin)

    plt.xlim(0, 0.00035)

    # Conductance plot
    for i in range(nranges):
        if i == 6:
            plt.ylim(0, 500000)
        rdata = pd_data[pd_data['bin'] == i]
        color = sns.color_palette()[i]
        sns.distplot(rdata['g'], kde=True, label='Range %d' % i, axlabel=True)
        plt.axvline(Manual_Gmins[i], 0, 1, color=color, linestyle=':', linewidth=1)
        plt.axvline(Manual_Gmaxs[i], 0, 1, color=color, linestyle=':', linewidth=1)
        plt.axvline(Write_Gmins[i], 0, 1, color=color, linestyle='-', linewidth=1)
        plt.axvline(Write_Gmaxs[i], 0, 1, color=color, linestyle='-', linewidth=1)
    plt.legend(borderpad=0.2, prop={'size': 10})
    plt.savefig("figs/" + filename.replace(".", "-").replace("/", "-") + ".pdf")
    # plt.show()


if __name__ == "__main__":
    filename = sys.argv[1]
    data_init()
    data_validate()
    draw()
