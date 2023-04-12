# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:30:10 2023

@author: rob
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import Graph_2 as G
import HITS_3 as H
import networkx as nx
import random
import copy
import math

        
        

def visualize_factordict(factordict):
    content = 1
    for factor, amount in factordict.items():
        content *= (factor ** amount)
    ID = content
    shape = (round(content ** 0.5), round(content ** 0.5))
    content = shape[0] * shape[1]
    sorted_factors = copy.deepcopy(list(factordict.keys()))
    sorted_factors.sort()

    
    
    available_space = content
    _1d = []
    while available_space > 0:
        rnd_float = 1 / random.random()
        #find the minimum distance and its index. thanks chatgpt
        closest = min(sorted_factors, key=lambda x: abs(x - rnd_float))
        for j in range(closest):
            _1d.append(1 / math.log(closest))
        available_space -= closest
    
    
    arr_1d = np.array(_1d[:content])
    arr_2d = arr_1d.reshape(shape)
    
    # Reverse every other row.
    '''for i in range(1, shape[0], 2):
        arr_2d[i] = arr_2d[i, ::-1]'''
            
    
    plt.rcParams['figure.dpi'] = 600
    
    ax = sns.heatmap(arr_2d, vmin=0.0, vmax=1/math.log(2), cbar_kws={'label': '1 / ln(factor)'})
    
    ax.set_xticks([])
    ax.set_yticks([])

    # add labels and title to the plot
    plt.title('Object: ' + str(ID) + ', Factors: ' + str(factordict))
    
    # display the plot
    plt.show() 
    
    
    
    
    
    


def visualize_query_to_root_set(query_factors, root_set):
    pass

    
    
def heatmap_node_user_adjacency_matrix(nodes, users):
    matrix = np.zeros((len(users), len(nodes)))
    for u in users:
        for c in u.children:
            matrix[u._id][c._id] = 1
            
    plt.rcParams['figure.dpi'] = 600
            
    ax = sns.heatmap(matrix, cbar_kws={'label': 'Connection Weights'})
    
    
    # add labels and title to the plot
    plt.xlabel('Nodes')
    plt.ylabel('Users')
    plt.title('Binary Adjacency Matrix of User-Node Connections')
    
    # display the plot
    plt.show()    
    

def draw_network_with_users(nodes, users):
    plt.rcParams['figure.dpi'] = 600
    nxGraph = G.add_users_to_nxGraph(nodes, users)
    plt.title("User-Connected Webgraph with " + str(nxGraph.number_of_nodes()) + " Nodes and " + str(nxGraph.number_of_edges()) + " Edges")
    nx.draw_networkx(nxGraph, with_labels=True)
    

def plot_avg_vs_user_trusts(user_trusts):
    plt.rcParams['figure.dpi'] = 600
    _x = np.arange(0, len(user_trusts), 1)
    # Create a scatter plot using seaborn
    sns.scatterplot(x=_x, y=user_trusts)

    
    # Add a linear regression line using seaborn
    plt.plot(_x, [0.5 for _ in range(len(user_trusts))], label='Average Network Trust')
    sns.regplot(x=_x, y=user_trusts, label='Average User Trust')
    
    
    # Add title and labels
    plt.title('Average Trust of the Network vs. Average Trust of User Connections')
    plt.xlabel('Iterations')
    plt.ylabel('Average Trust')
    
    plt.legend()
    
    # Show the plot
    plt.show()


def heatmap_trusts(trusts, names):
    
    plt.rcParams['figure.dpi'] = 600
    
    # create a heatmap using Seaborn
    ax = sns.heatmap(trusts, cbar_kws={'label': 'Trust Values'})
    
    ax.set_yticklabels(names, rotation=0)
    
    # add labels and title to the plot
    plt.xlabel('Nodes')
    plt.title('Trust Comparison')
    
    # display the plot
    plt.show()

def heatmap_adjacency_matrix(nodes):
    matrix = np.zeros((len(nodes), len(nodes)))
    for n in nodes:
        for k, v in n.edges.items():
            matrix[n._id][k] = v
            
    plt.rcParams['figure.dpi'] = 600
            
    ax = sns.heatmap(matrix, cbar_kws={'label': 'Edge Weights'})
    
    
    # add labels and title to the plot
    plt.xlabel('Children')
    plt.ylabel('Parents')
    plt.title('Adjacency Matrix')
    
    # display the plot
    plt.show()
        


def heatmap_auth_rankings(auths, names):
    plt.rcParams['figure.dpi'] = 600
    # create a heatmap using Seaborn
    ax = sns.heatmap(auths, annot=True, cbar_kws={'label': 'Node IDs'})
    
    ax.set_yticklabels(names, rotation=0)
    
    # add labels and title to the plot
    plt.xlabel('Authority Rank')
    plt.title('Authority Value Ranking from Lowest to Highest')
    
    # display the plot
    plt.show()
    
    
def heatmap_hub_rankings(hubs, names):
    plt.rcParams['figure.dpi'] = 600
    # create a heatmap using Seaborn
    ax = sns.heatmap(hubs, annot=True, cbar_kws={'label': 'Node IDs'})
    
    ax.set_yticklabels(names, rotation=0)
    
    # add labels and title to the plot
    plt.xlabel('Hub Rank')
    plt.title('Hub Value Ranking from Lowest to Highest')
    
    # display the plot
    plt.show()
    
    
    
if __name__ == "__main__":
    
    fd = {2:2, 5:3, 3:4}
    fd1 = {2:3, 3:1, 5:1}
    fd2 = {2:1, 7:1, 13:1, 27:1}
    fd3 = {13: 1}
    
    print(fd)
    
    #visualize_factordict(fd)
    visualize_factordict(fd1)
    #visualize_factordict(fd2)

    visualize_factordict(H.prime_factors_to_dict(H.prime_factors(random.randint(2, 100000))))
    visualize_factordict(H.prime_factors_to_dict(H.prime_factors(random.randint(2, 100000))))
    visualize_factordict(H.prime_factors_to_dict(H.prime_factors(random.randint(2, 100000))))

    
    
 
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





