# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:39:51 2023 

@author: rob
"""

import matplotlib.pyplot as plt
import numpy as np

def is_integer(x):
    return x == int(x)

def trial_division(n):
    _n = n
    factors = []
    factor = 2
    n_steps = 1
    
    
    while factor*factor <= _n:
        if _n % factor == 0:
            factors.append(factor)
            _n /= factor
        else:
            factor += 1        
        n_steps += 1
        
    if _n != 1:
        factors.append(_n)
            
    return factors, n_steps

def trial_division_with_information(n, known_factors):
    _n = n
    for known_factor in known_factors:
        _n /= known_factor
        if not is_integer(_n):
            return trial_division(n)
    factors, n_steps = trial_division(_n)
    n_steps += len(known_factors)
    return factors, n_steps








fs = []
ss = []
lenfs = []
rng = 100000
sms = []
ratios = []
values = []
for i in range(2, 2+rng):
    f, s = trial_division(i)
    fs.append(f)
    ss.append(s)
    lenfs.append(len(f))
    su = sum(f)
    sms.append(s)
    ratio = ((i/s) / np.sqrt(i))
    ratios.append(ratio)
    value = np.log(i)*ratio
    values.append(value)
    
    print(value, ratio,su, f, s, i)
    

    

x = np.linspace(0, rng, rng)
plt.scatter(x, ratios)
plt.show()


            






