import numpy as np

mini = 0
maxi = 0
bins = 0
interval = 0
sigma = {}
timestamps = []

def load_param():
    global mini, maxi, bins, interval, timestamps
    with open("conf", "r") as fin:
        lines = fin.readlines()
        mini, maxi, bins = lines[0].split(",")
        mini, maxi, bins = float(mini), float(maxi), int(bins)
        interval = (maxi - mini) / bins
        for i in range(1, len(lines)):
            t, idx, sig = lines[i].split(",")
            t, idx, sig = float(t), int(idx), float(sig)
            if t not in sigma.keys():
                sigma[t] = {}
            sigma[t][idx] = sig
        timestamps = sorted(list(sigma.keys()))

def get_sigma(g0, t):
    def get_bin(g):
        assert mini <= g and g <= maxi
        idx = (g - mini) / interval
        assert int(idx) < bins
        return int(idx)

    def get_tidx_weights(stmp):
        for i in range(len(timestamps) - 1):
            if timestamps[i] <= stmp and stmp < timestamps[i+1]:
                weight1 = (timestamps[i+1] - stmp) / (timestamps[i+1] - timestamps[i])
                weight2 = (stmp - timestamps[i]) / (timestamps[i+1] - timestamps[i])
                return i, weight1, weight2

    idx = get_bin(g0)
    if t in timestamps:
        print(t, idx)
        return sigma[t][idx]
    elif t < timestamps[0]:
        return sigma[timestamps[0]][idx]
    elif t > timestamps[0]:
        return sigma[timestamps[-1]][idx]
    else:
        tidx, w1, w2 = get_tidx_weights(timestamps)
        return w1 * sigma[timestamps[tidx]][idx] + w2 * sigma[timestamps[tidx+1]][idx]

def drift(g0, t):
    sig = get_sigma(g0, t)
    diff = np.random.normal(0, sig)
    print(g0 + diff)


if __name__ == "__main__":
    load_param()
    g0 = float(input("Write g0 between " + str(mini) + ", " + str(maxi) + "\n"))
    drift(g0, 0.1)
