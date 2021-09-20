import matplotlib.pyplot as plt
# names = ['addr', 'nreads', 'nsets', 'nresets', 'rf', 'if', 'rlo', 'rhi', 'success', 'attempts1', 'attempts2']

# 0 addr
# 4 rf (read final)
# 6 rlo (range low)
# 7 rhi (range high)
# 8 success

data = {}
length = 0
error_rate = 0.0
low_high = {}

def data_init():
    def add_to_dict(dct, entry_name, init_value, append_item):
        if entry_name not in dct.keys():
            dct[entry_name] = init_value
        dct[entry_name].append(append_item)
    with open("../data/sdr-4wl-eval-chip2-8k-8-9-20.csv", "r") as fin:
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
        error_rate = data["success"].count(0) / length
        low_high = dict(zip(data["rlo"], data["rhi"])).items()
    
    def print_features():
        print("length", length)
        print("error_rate", error_rate)
        print("low_high", low_high)

    def validate_write_success(low, high, final, success):
        if final > low and final < high:
            assert success == 1
        else:
            assert success == 0
    
    extract_features()
    print_features()
    for i in range(length):
        validate_write_success(data["rlo"][i], data["rhi"][i], data["rf"][i], data["success"][i])

def draw():
    x = data["rf"]
    plt.hist(x, bins=length)
    plt.show()

if __name__ == "__main__":
    data_init()
    data_validate()
    draw()
    # print(data)