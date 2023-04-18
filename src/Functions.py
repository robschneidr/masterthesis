# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:26:43 2023

@author: rob
"""

import math
import numpy as np
import matplotlib.pyplot as plt

#inverse_logistic_false_factor_difference
def ilffd(y):
    return 0.5 - 2 * math.log(1 / y - 1)
#logistic_false_factor_difference
def lffd(x):
    return 1 / (1 + math.exp(-0.5 * (x - 0.5)))


def false_factor_probability_function(x):
    return 1 - (1 / math.exp(3 * x))

def inverse_false_factor_probability_function(y):
    return -(1 / 3) * math.log(1 - y, math.e)    



if __name__ == "__main__":

    x = np.arange(-5, 5, 0.05)     
    #y = [false_factor_difference_function(_x) for _x in x]
    y = [lffd(_x) for _x in x]
    z = [ilffd(_z) for _z in y]
    
    print([(a, b) for (a, b) in zip(x, y)])
    
    plt.plot(x, y)
    #plt.plot(x, z)
    print([(a, b) for (a, b) in zip(y, z)])
    plt.show()
    
    