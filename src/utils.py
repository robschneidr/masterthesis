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

    # Generate data for the initial distributions
    np.random.seed(0)
    initial_distributions = [
        np.random.normal(loc=0, scale=1, size=1000),
        np.random.uniform(low=-1, high=1, size=1000),
        np.random.exponential(scale=1, size=1000),
        np.random.gamma(shape=2, scale=1, size=1000),
        np.random.lognormal(mean=0, sigma=1, size=1000)
    ]
    
    # Generate data for the target distributions
    np.random.seed(1)
    target_distributions = [
        np.random.normal(loc=3, scale=1.5, size=1000),
        np.random.uniform(low=2, high=4, size=1000),
        np.random.exponential(scale=2, size=1000),
        np.random.gamma(shape=4, scale=0.5, size=1000),
        np.random.lognormal(mean=1, sigma=1, size=1000)
    ]
    
    # Set up the transition steps (e.g., 5 steps)
    transition_steps = 5
    
    # Compute intermediate distributions
    distributions = []
    for initial, target in zip(initial_distributions, target_distributions):
        intermediate_distributions = []
        for step in range(transition_steps + 1):
            intermediate_distribution = (
                initial * (transition_steps - step) / transition_steps
                + target * step / transition_steps
            )
            intermediate_distributions.append(intermediate_distribution)
        distributions.append(intermediate_distributions)
    
    # Plot the transition using histograms
    fig, axes = plt.subplots(5, 5, figsize=(15, 12))
    
    row_titles = ['Distribution 1', 'Distribution 2', 'Distribution 3', 'Distribution 4', 'Distribution 5']
    
    for row, intermediate_distributions in enumerate(distributions):
        for col, distribution in enumerate(intermediate_distributions[:5]):
            sns.histplot(distribution, kde=True, ax=axes[row, col], color='skyblue')
            axes[row, col].set_title(f'Step {col*5}/{20}')
            axes[row, col].set_xlabel('')
            axes[row, col].set_ylabel('')
            axes[row, col].set_xlim(-5, 10)  # Adjust x-axis limits if needed
    
        axes[row, 0].set_ylabel(row_titles[row])
    

    
    plt.tight_layout()
    plt.show()
