import drift_model
import matplotlib.pyplot as plt
import seaborn as sns

class Level(object):
    '''
    Level is represented as Read Range [r1, r2] and Write Range [w1, w2]
    [w1, w2] should be within the range of [r1, r2] 
    '''
    def __init__(self, r1, r2, w1, w2, sigma=0, prob=0):
        assert r1 < w1 and w1 < w2 and w2 < r2
        self.r1 = r1
        self.r2 = r2
        self.w1 = w1
        self.w2 = w2
        self.sigma = sigma
        self.prob = prob
    
    def __str__(self):
        return "Read:[%d,%d], Write:[%d,%d]" % (self.r1, self.r2, self.w1, self.w2)
    
    def draw(levels):
        for i in range(len(levels)):
            color = sns.color_palette(n_colors=len(levels))[i]
            plt.axvline(levels[i].r1, color=color, linestyle=':', linewidth=1)
            plt.axvline(levels[i].r2, color=color, linestyle=':', linewidth=1)
            plt.axvline(levels[i].w1, color=color, linestyle='-', linewidth=1)
            plt.axvline(levels[i].w2, color=color, linestyle='-', linewidth=1)
        plt.show()

    @staticmethod
    def overlap(A, B) -> bool:
        if B.r2 >= A.r1 and A.r2 >= B.r1:
            return True
        else:
            return False
    
    @staticmethod
    def sort_by_mean(all_levels):
        return sorted(all_levels, key=lambda x: (x.w1 + x.w2) / 2)
    
    @staticmethod
    def longest_non_overlap(all_levels):
        '''
        This is inaccurate greedy algorithm
        Assumption:
            interval of read ranges increases with the resistance
        '''
        res = []
        sorted_levels = Level.sort_by_mean(all_levels)
        res.append(sorted_levels[0])
        cur = sorted_levels[0]
        for i in range(1, len(sorted_levels)):
            nxt = sorted_levels[i]
            if Level.overlap(cur, nxt) == False:
                res.append(nxt)
                cur = nxt
        return res


def find_densest_repr(Rmin, Rmax, prob, write_width, exact_width=True):
    '''
    Arguments:
        Rmin, Rmax: Minimum/Maximum of the resistance range
        prob: probability specification of bit to be written/read correctly
        [missing] n: the number of levels [do not necessarily need to be 2^n]
        write_width: the write width of the write range (w2-w1), can be either minimum or exact
        exact_width: boolean value. Default is true, then no need to search w2 (high write resistance)

    Return:
        The densest representation (highest n)
    
    Implementation:
        TODO: exact_width not considered
        Simple greedy algorithm, so n not used for now
        Enumerate all possible R_{center} by trying all i for [R_{min} + i*delta]
        Because exact_width=True, and write_width is provided, so [w1, w2] is determined
        Then invoke prob2read with write range and prob as params -> get read range
        Save all the potential Level computed above (all_levels)
        [So far, n not used (due to greedy algorithm)]
        Finally compute the longest non_overlapping levels (to get densest repr)
    '''
    delta = 10
    all_levels = []
    i, Rcenter = 1, Rmin
    while True:
        Rcenter = Rmin + i * delta
        w1, w2 = Rcenter - write_width / 2, Rcenter + write_width / 2
        if w2 >= Rmax:
            break
        sigma = drift_model.get_max_sigma(w1, w2)
        r1, r2 = drift_model.prob2read(w1, w2, prob)
        all_levels.append(Level(r1, r2, w1, w2, sigma, prob))
        i += 1
    return Level.longest_non_overlap(all_levels)

    

if __name__ == "__main__":
    drift_model.load_param()
    prob = 0.99
    write_width = 100
    print(drift_model.mini, drift_model.maxi)
    mini = drift_model.mini
    maxi = drift_model.maxi
    solutions = find_densest_repr(drift_model.mini, maxi, prob, write_width)
    # Level.draw(solutions)
    print(len(solutions))
