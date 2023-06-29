# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:04:28 2023

@author: rob
"""


import numpy as np
import random
import Graph_5 as G
import HITS_5 as H
import matplotlib.pyplot as plt
import math
import seaborn as sns
from collections import deque
import pandas as pd
from scipy.stats import norm

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
        
        

def plot_equivalence_points_distribution():

    b = 9
    boxes = np.zeros(b)
    n = 2000

    for _ in range(n):
    
        idA = random.randint(2, 500000000)
        idB = random.randint(2, 500)    
        
        factorsA = H.prime_factors(idA)
        factorsB = H.prime_factors(idB)

        dA = H.prime_factors_to_dict(factorsA)
        dB = H.prime_factors_to_dict(factorsB)

        c = H.compare_semantic_equivalence(dA, dB)
        
        boxes[c] += 1
        


    plt.bar(range(b), boxes)
    plt.show()      




if __name__ == "__main__":
    

  
    x = [1, 2, 3, 4, 5]  # X-axis values
    y1 = [10, 15, 7, 12, 9]  # First y-axis values
    y2 = [900, 350, 50, 420, 70]  # Second y-axis values
    
    fig, ax1 = plt.subplots()  # First subplot with left y-axis
    ax2 = ax1.twinx()  # Second subplot with right y-axis
    
    sns.regplot(x=x, y=y1, ax=ax1, color='g', label='Y1')  # Regression plot for the first y-axis
    sns.regplot(x=x, y=y2, ax=ax2, color='b', label='Y2')  # Regression plot for the second y-axis
    
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y1-axis', color='g')
    ax2.set_ylabel('Y2-axis', color='b')
    
    ax1.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='b')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.show()

    
    




    


     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    