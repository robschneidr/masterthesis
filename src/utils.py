# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:04:28 2023

@author: rob
"""


import numpy as np
import Graph



def dothis():
    return 0

def dothat():
    return 1

def setA(ana, func):
    ana.func()

class A:
    
    def __init__(self):
        self.c = 2
        
    def do1(self):
        self.c = 5



if __name__ == "__main__":
    
    ana = A()
    print(ana.c)
    
    #setA(ana, dothat)
    setA(ana, ana.do1)
    print(ana.c)
    
    
    
    