# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:04:28 2023

@author: rob
"""


import numpy as np
import random
import Graph_3 as G
import HITS_3 as H
import matplotlib.pyplot as plt

def mean_nodes_order_similarity(nodeIDs_A, nodeIDs_B):
    
     switched_order_A = [0 for _ in range(len(nodeIDs_A))]
     switched_order_B = [0 for _ in range(len(nodeIDs_B))]
     
     print(nodeIDs_A)
     print(nodeIDs_B)
     
     for i in range(len(nodeIDs_A)):
         switched_order_A[nodeIDs_A[i]] = i
         switched_order_B[nodeIDs_B[i]] = i
         
     print(switched_order_A)
     print(switched_order_B)
     
     sum_differences = 0    
     for A, B in zip(switched_order_A, switched_order_B):
         sum_differences += abs(A - B)
     sum_differences /= len(switched_order_A)
     
     return sum_differences



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
    

    node = G.Node(5)
    
    G.set_content_and_private_factors(node, 10000)
    print(node.content)
    print(node.private_factors)
    
    G.set_public_factors(node, 1.2)
    
    print(node.public_factors)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    