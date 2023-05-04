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
import seaborn as sns
from collections import deque
import pandas as pd

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
    

    # create some sample time series data
    data = pd.Series(data=np.random.randn(2500), index=pd.date_range(start='2022-01-01', periods=2500, freq='D'))
    
    # create the line plot with markers
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data.values, linewidth=1, color='blue', marker='o', markersize=3)
    
    # set the plot title and axes labels
    ax.set_title('Time Series Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    
    # rotate x-axis labels for better visibility
    fig.autofmt_xdate()
    
    # show the plot
    plt.show()
    
    




    


     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    