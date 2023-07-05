# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:06:25 2023

@author: rob
"""

import Graph_1_2 as G
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import visualization as v
import sys

name_id = 5

  

def set_edge_trusts(nodes, val=-1):
    for n in nodes:
        for child, weight in n.edges.items():
            if val < 0:
                n.edges.update({child : random.random()})
            else:
                n.edges.update({child : val})
            
           

def set_all_auths(nodes, auths):
    for node, auth in zip(nodes, auths):
        node.auth = auth
    normalize_auths(nodes, sum(auths))
       
def normalize_auths(nodes, sum_auths):
    for node in nodes:
        node.auth /= sum_auths
                
        
def set_all_hubs(nodes, hubs):
    for node, hub in zip(nodes, hubs):
        node.hub = hub
    normalize_hubs(nodes, sum(hubs))  
        
def normalize_hubs(nodes, sum_hubs):
    for node in nodes:
        node.hub /= sum_hubs
        
        
def set_all_trusts(nodes, trusts):
    for node, trust in zip(nodes, trusts):
        node.trust = trust
    #normalize_trusts(graph, sum(trusts))
        
def normalize_trusts(nodes, sum_trusts):
    for node in nodes:
        node.trust /= sum_trusts
        

    
    
def HITS_iteration(nodes, 
                   std_HITS, 
                   trust_normalized, 
                   trust_belief, 
                   trust_node, 
                   trust_connection, 
                   auth_distribution):
    
       
    for i in range(n_steps):
        HITS_one_step(nodes, std_HITS, trust_normalized, trust_belief, trust_node, trust_connection)
        
        if i % 5 == 0:
            auth_distribution.append([n.auth for n in nodes])
        #print_hubAuthTrust_values(nodes)
    
    print_hubAuthTrust_values(nodes)
    #print_parents_children(nodes)
    
    return nodes
        
        
        
        

def HITS_one_step(nodes, std_HITS, trust_normalized, trust_belief, trust_node, trust_connection):

    
    auths_old = dict()
    hubs_old = dict()
    trusts_old = dict()
    edges_old = []
    
    for n in nodes:
        auths_old[n._id] = n.auth
        hubs_old[n._id] = n.hub
        trusts_old[n._id] = n.trust
        edges_old.append(n.edges)
        
    for node in nodes:
        
        parents = [n._id for n in node.parents]
        children = [n._id for n in node.children]
        
        if std_HITS:
            node.auth = sum(hubs_old[p] for p in parents)
            node.hub = sum(auths_old[c] for c in children) 
            
        if trust_normalized and trust_node:
            node.auth = sum(hubs_old[p] + trusts_old[p] for p in parents)
            node.hub = sum(auths_old[c] + trusts_old[c] for c in children)
            
        if trust_normalized and trust_connection:
            node.auth = sum(hubs_old[p] + edges_old[p].get(node._id) for p in parents)
            node.hub = sum(auths_old[c] + edges_old[node._id].get(c) for c in children)
            
        if trust_belief and trust_node:                
            node.auth = sum(hubs_old[p] * trusts_old[p] for p in parents)
            node.hub = sum(auths_old[c] * trusts_old[c] for c in children)
            
        if trust_belief and trust_connection:
            node.auth = sum(hubs_old[p] * edges_old[p].get(node._id) for p in parents)
            node.hub = sum(auths_old[c] * edges_old[node._id].get(c) for c in children)
        

    normalize_auths(nodes, sum(n.auth for n in nodes))
    normalize_hubs(nodes, sum(n.hub for n in nodes))
        
    return nodes

        

def switch_nodes_order(nodeIDs):
    switched_orders = []
    for n in nodeIDs:
        switched_orders.append([0 for _ in range(len(n))])

    
    for n, sw in zip(nodeIDs, switched_orders):
        for i in range(len(n)):
            sw[n[i]] = i
        
    return switched_orders

def mean_nodes_order_similarity(nodeIDs_A, nodeIDs_B):
    
     switched_order_A = [0 for _ in range(len(nodeIDs_A))]
     switched_order_B = [0 for _ in range(len(nodeIDs_B))]
     
     #print(nodeIDs_A)
     #print(nodeIDs_B)
     
     for i in range(len(nodeIDs_A)):
         switched_order_A[nodeIDs_A[i]] = i
         switched_order_B[nodeIDs_B[i]] = i
         
     #print(switched_order_A)
     #print(switched_order_B)
     
     sum_differences = 0    
     for A, B in zip(switched_order_A, switched_order_B):
         sum_differences += abs(A - B)
     sum_differences /= len(switched_order_A)
     
     return sum_differences
     

    
def get_sorted_nodes(nodes):
    
    sorted_nodes_auth = copy.copy(nodes)
    sorted_nodes_auth.sort(key=sort_by_auth)
    
    sorted_nodes_hub = copy.copy(nodes)
    sorted_nodes_hub.sort(key=sort_by_hub) 

    sorted_nodes_trust = copy.copy(nodes)
    sorted_nodes_trust.sort(key=sort_by_trust)
    
    return sorted_nodes_auth, sorted_nodes_hub, sorted_nodes_trust
    
    
def sort_by_trust(node):
    return node.trust

def sort_by_auth(node):
    return node.auth

def sort_by_hub(node):
    return node.hub

def get_trust_values(params):
    return [[node.trust for node in p] for p in params]
        

def get_sorted_nodeIDs(params, sorted_nodes_auths, sorted_nodes_hubs, sorted_nodes_trusts):
    
    sorted_nodeIDs_auths = []
    sorted_nodeIDs_hubs = []
    sorted_nodeIDs_trusts = []
    
    for i in range(3):
        if i == 0:
            value_name = "auth"
            vals = sorted_nodes_auths
            sorted_nodeIDs = sorted_nodeIDs_auths
            
        elif i == 1:
            value_name = "hub"
            vals = sorted_nodes_hubs
            sorted_nodeIDs = sorted_nodeIDs_hubs
        else:
            value_name = "trust"
            vals = sorted_nodes_trusts
            sorted_nodeIDs = sorted_nodeIDs_trusts
                      
        for i, param in enumerate(params):
            sorted_nodeIDs.append([node._id for node in vals[i]])
            #print([node._id for node in vals[i]], "node " + value_name + " order " + param[name_id])
            
        print()
        
    print()
            
    return sorted_nodeIDs_auths, sorted_nodeIDs_hubs, sorted_nodeIDs_trusts



    
    



'''_______________________________UTILS_____________________________________'''  

def set_params(*args):
    return [arg for arg in args] 
        

def print_hubAuthTrust_values(nodes):
    for n in nodes:
        print("node", n._id, " hub = ", n.hub, "   auth =", n.auth, " node_trust = ", n.trust)
    print()

def print_parents_children(nodes):
    for n in nodes:
        print("node ", n._id, "parents: ", [p._id for p in n.parents], ", children: ", [c._id for c in n.children], "edges: ", [(e, v) for e, v in n.edges.items()])   
        
    print()


def plot_node_rankings(params, sorted_nodeIDs_auths, sorted_nodeIDs_hubs, sorted_nodeIDs_trusts):
    x = np.arange(0, len(sorted_nodeIDs_auths[0]))   
    fig, axs = plt.subplots(3, 1)
    for ID, param in zip(sorted_nodeIDs_auths, params): 
        #print(ID, param[name_id])
        axs[0].plot(x, ID, label=param[name_id])

    axs[0].set_ylabel('Authority')
    #axs[0].grid(True)
    
    for ID, param in zip(sorted_nodeIDs_hubs, params): 
        #print(ID, param[name_id])
        axs[1].plot(x, ID, label=param[name_id])
    axs[1].set_ylabel('Hub')
    #axs[1].grid(True)
    
    for ID, param in zip(sorted_nodeIDs_trusts, params): 
        #print(ID, param[name_id])
        axs[2].plot(x, ID, label=param[name_id])
    axs[2].set_xlabel('Node Ranking from Lowest to Highest')
    axs[2].set_ylabel('Trust')
    #axs[2].grid(True)
    
    fig.tight_layout()
    plt.legend(loc='best', bbox_to_anchor=(1.2, 1))
    plt.show()
    
    
'''_______________________________UTILS_____________________________________'''
    


if __name__ == '__main__':
    
   
    
    n_steps = 21
    n_nodes = 20
    n_edges = 80
 
    
    std_HITS = [True, False, False, False, False, "Standard HITS"]
    trust_normalized_node = [False, True, False, True, False, "Normalized Node Trust"]
    trust_normalized_connection = [False, True, False, False, True, "Normalized Connection Trust"]
    trust_belief_node = [False, False, True, True, False, "Belief Node Trust"]
    trust_belief_connection = [False, False, True, False, True, "Belief Connection Trust"]
    

    
    
    #base_nodes = G.create_random_weighted_directed_document_nodes(n_nodes, n_edges)
    base_nodes = G.load_graph("rnd_20n80e_2")
    G.visualize(base_nodes)
    print_parents_children(base_nodes)
    #G.save_graph(base_nodes, "rnd_20n80e_2")
    
    params = [std_HITS, 
              trust_normalized_node, 
              trust_normalized_connection,
              trust_belief_node,
              trust_belief_connection]
    
    #params = [params[0], params[-1]]
    #params = [params[2]]
    #params = [params[0], params[2], params[5], params[7], params[8]]
    #params = [params[0]]
    
    node_copies = [G.create_nodes_copy(base_nodes) for nodes in range(len(params))]
    
    
    sorted_nodes_auths = []
    sorted_nodes_hubs = []
    sorted_nodes_trusts = []
    unsorted_nodes = []
    auth_distributions = [[] for _ in range(len(params))]
    
    
    for nodes, param, auth_distribution in zip(node_copies, params, auth_distributions):
        
        print("\n\n", param[name_id], "\n")

        
        HITS_iteration(nodes,
                       param[0],
                       param[1],
                       param[2],
                       param[3],
                       param[4],                       
                       auth_distribution)
        

        
        unsorted_nodes.append(nodes)
        sorted_nodes_auth, sorted_nodes_hub, sorted_nodes_trust = get_sorted_nodes(nodes)
        sorted_nodes_auths.append(sorted_nodes_auth)
        sorted_nodes_hubs.append(sorted_nodes_hub)
        sorted_nodes_trusts.append(sorted_nodes_trust)
        
      
    
    
    
    
    
    sorted_nodeIDs_auths, sorted_nodeIDs_hubs, sorted_nodeIDs_trusts = get_sorted_nodeIDs(params, sorted_nodes_auths, sorted_nodes_hubs, sorted_nodes_trusts)    
    
    mean_order_similarities = []
    for sorted_nodeIDs_auth, param in zip(sorted_nodeIDs_auths, params):
        mean_order_similarities.append((mean_nodes_order_similarity(sorted_nodeIDs_auths[0], sorted_nodeIDs_auth), param[name_id]))
    print(mean_order_similarities)
    
    '''for row in auth_distributions:
        for col in row:
            print(col)
        print()'''
    v.plot_auth_distribution_transitions(auth_distributions, [param[name_id] for param in params])
    v.heatmap_auth_rankings(sorted_nodeIDs_auths, [param[name_id] for param in params])
    v.heatmap_hub_rankings(sorted_nodeIDs_hubs, [param[name_id] for param in params])
    #v.heatmap_adjacency_matrix(nodes)
    #v.heatmap_trusts(get_trust_values(unsorted_nodes[5:7]), [param[name_id] for param in params[5:7]])
    #v.heatmap_trusts(get_trust_values(unsorted_nodes), [param[name_id] for param in params])
    
    
    #plot_node_rankings(params, sorted_nodeIDs_auths, sorted_nodeIDs_hubs, sorted_nodeIDs_trusts)        
    #diff = mean_nodes_order_similarity([node._id for node in sorted_nodes_auths[0]], [node._id for node in sorted_nodes_auths[1]])
    #print(diff)    
        
    
    
       
        
        
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    