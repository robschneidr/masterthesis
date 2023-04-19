# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:26:43 2023

@author: rob
"""

import math
import numpy as np
import matplotlib.pyplot as plt

INVERSE_EXP_CUTOFF = 10
K_CUTOFF = 200
LOG_CEIL = 0.999999
LOG_FLOOR = 0.000001


# ranking value to false factor probability
def rankingValue_to_falseFactorProbability(x):
    return 1 / (1 + math.exp(-x))

#false factor probability to ranking value
def falseFactorProbability_to_rankingValue(y):
    _y = min(LOG_CEIL, max(LOG_FLOOR, y))
    return -math.log((1 / _y) - 1)
    
#this models the belief that the "first" site of google results is by far the most important
#ranking position to relative ranking value
def rankingPosition_to_rankingValue(x, root_set_size):
    k = K_CUTOFF ** (1 / root_set_size) - 1
    return 1 / math.pow(1 + k, x)


#false factor value to trust probability
def falseFactorValue_to_trustProbability(x):
    return 1 / (1 + math.exp(-x))

#trust probability to false factor value
def trustProbability_to_falseFactorValue(y):
    _y = min(LOG_CEIL, max(LOG_FLOOR, y))
    return -math.log((1 / _y) - 1)

  



if __name__ == "__main__":

    '''x = np.arange(-5, 5, 0.05)     
    #y = [false_factor_difference_function(_x) for _x in x]
    y = [lffd(_x) for _x in x]
    z = [ilffd(_z) for _z in y]
    
    print([(a, b) for (a, b) in zip(x, y)])
    
    plt.plot(x, y)
    #plt.plot(x, z)
    print([(a, b) for (a, b) in zip(y, z)])
    plt.show()'''
    
    a = np.arange(1, 40)
    b = [rankingPosition_to_rankingValue(_a, 39) for _a in a]
    
    print(b)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    