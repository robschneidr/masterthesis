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
import math

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


def test_get_private_ranking():
    search_factors = [2, 2, 5]
    n_nodes = 30
    nodes = []
    
    for i in range(n_nodes):
        nodes.append(G.Node(i))
        G.set_content_and_private_factors(nodes[i], 1000000)
        print(i, ": ", nodes[i].private_factors)
    
    
    
    print(H.get_private_ranking(nodes, H.prime_factors_to_dict(search_factors)))
        
        

        




if __name__ == "__main__":
    #test_get_private_ranking()
    
    a = 0.3
    b = H.false_factor_probability_function(a)
    print(b)
    
    c = H.inverse_false_factor_probability_function(b)
    print(c)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    