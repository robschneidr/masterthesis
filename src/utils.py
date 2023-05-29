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
    

  

    # Generate example datasets
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(2, 0.5, 1000)
    data3 = np.random.normal(-2, 1.5, 1000)
    
    # Fit a normal distribution to each dataset
    params1 = norm.fit(data1)
    params2 = norm.fit(data2)
    params3 = norm.fit(data3)
    
    # Plot histograms for each dataset
    plt.hist(data1, bins=30, density=True, alpha=0.5, label='Dataset 1')
    plt.hist(data2, bins=30, density=True, alpha=0.5, label='Dataset 2')
    plt.hist(data3, bins=30, density=True, alpha=0.5, label='Dataset 3')
    
    # Plot estimated normal distribution curves
    x = np.linspace(-6, 6, 1000)
    pdf1 = norm.pdf(x, params1[0], params1[1])
    pdf2 = norm.pdf(x, params2[0], params2[1])
    pdf3 = norm.pdf(x, params3[0], params3[1])
    
    plt.plot(x, pdf1, color='red', linewidth=2, label='Normal Distribution (Dataset 1)')
    plt.plot(x, pdf2, color='blue', linewidth=2, label='Normal Distribution (Dataset 2)')
    plt.plot(x, pdf3, color='green', linewidth=2, label='Normal Distribution (Dataset 3)')
    
    # Add legend and labels
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    
    # Display the plot
    plt.show()

    
    




    


     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    