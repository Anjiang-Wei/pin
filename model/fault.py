import torch
import random
from bitstring import BitArray

first = True
sum_diff = 0

def random_oracle_float():
    res = []
    # for i in range(0, 9):
    #     pass
    for i in range(9, 32):
        if random.randint(0, 1) == 1:
            res.append(i)
    return res


def mutate_float(fval):
    global sum_diff
    global first
    f1 = BitArray(float=fval, length=32)
    # binary = f1.bin
    f1.invert(random_oracle_float())
    fval2 = f1.float
    if first:
        # print("difference: ", fval2 - fval)
        first = True
        sum_diff += abs(fval2 - fval)
    return fval2
    

def mutate_list(vals):
    res = []
    for v in vals:
        if type(v) == float:
            v_ = mutate_float(v)
            res.append(v_)
        else:
            raise NotImplementedError
    return res


def mutate_tensor(tensor):
    global sum_diff
    sum_diff = 0
    size = list(tensor.size())
    flatten = torch.flatten(tensor)
    mutated_list = mutate_list(flatten.tolist())
    new_tensor = torch.tensor(mutated_list, dtype=tensor.dtype)
    new_tensor = new_tensor.view(*size)
    print("sum_diff:", tensor.size(), sum_diff)
    return new_tensor
