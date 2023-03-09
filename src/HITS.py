# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:06:25 2023

@author: rob
"""

import Graph
import random
import copy
import numpy as np
import matplotlib.pyplot as plt



'''
insights:
    
    fixed initial trust values do not make sense,
    because trust just becomes a measure of overall connectedness.
    
    this also applies to the hub and authorities values. if uniformly
    initialized at 1.0, they solely become a measure of connectedness...
    
    the deciding factor is indeed the prior requried for the initialization
    
    ==> the prior is actually not the deciding factor. see node trust order
    
    
    01.03. change the trust mechanism to calculate average and no normalized
    observation: the average trust in the network remains the same(?) during
    the iterations
    setting(1, 1, -1):
    trust values  average out to a common value
    
    
    TODO: make statistics about how different trust mechanisms alter auth and hub orders
    
    
    06.03.
    specifications for expose:
        1. show hits converges to the same values regardless of rnd or 1.0 initialization
        2. trust mechanics:
            show that trust is able to alter the hits ranking
            2 different versions of trust:
                1. absolute value --> normalized similar to hits
                2. average value --> classic average
                    -> note that trust can not be determined by the node itself, but only
                    by other nodes in connection with the node (a king who calls himself a king)
        
    
    structure of the webgraph:
        internal vs external links
        https://www.contentpowered.com/blog/many-external-links-articles/
        avg somewhere around 15 external links, 100 internal links
        build webgraph accordingly
        problem: there are 1.8 billion websites on the internet, each of which has
        15 external and 100 internal links. this is not realistically simulatable
        on very small scale... with 20 nodes eg, the graph would almost be fully connected
        
        
        
    user inclusion:
        a user is a node in the network that only has outgoing connections
        
        
        
    09.03.
    
    instead of deepcopying the auth hub trust values, perhaps it is better to deepcopy the graph itself for each
    parameter setting
    
    considerations on trustworthiness: right now the trustworthiness between a connection from A to B is determined
    solely by the trustworthiness of B. how about "individual trustworthiness" of each connection. isnt this a more
    realistic concept since each node has an individual trust rating of the connection to another node?
    
    

'''


def HITS_init(graph, auth=-1, hub=-1, trust=-1):
    
        
    if auth >= 0:
        auths = [auth for i in range(len(graph.nodes))]
        set_all_auths(graph, auths)
    else:
        rnd_auths = [random.random() for i in range(len(graph.nodes))]
        set_all_auths(graph, rnd_auths)
        
    if hub >= 0:
        hubs = [hub for i in range(len(graph.nodes))]
        set_all_hubs(graph, hubs)
    else:
        rnd_hubs = [random.random() for i in range(len(graph.nodes))]
        set_all_hubs(graph, rnd_hubs)    
    
    if trust >= 0:
        trusts = [trust for i in range(len(graph.nodes))]
        set_all_trusts(graph, trusts)
    else:
        rnd_trusts = [random.random() for i in range(len(graph.nodes))]
        set_all_trusts(graph, rnd_trusts)
        

def set_all_auths(graph, auths):
    for node, auth in zip(graph.nodes, auths):
        node.auth = auth
    normalize_auths(graph, sum(auths))
       
def normalize_auths(graph, sum_auths):
    for node in graph.nodes:
        node.auth /= sum_auths
        
        
def set_all_hubs(graph, hubs):
    for node, hub in zip(graph.nodes, hubs):
        node.hub = hub
    normalize_hubs(graph, sum(hubs))  
        
def normalize_hubs(graph, sum_hubs):
    for node in graph.nodes:
        node.hub /= sum_hubs
        
        
def set_all_trusts(graph, trusts):
    for node, trust in zip(graph.nodes, trusts):
        node.trust = trust
    #normalize_trusts(graph, sum(trusts))
        
def normalize_trusts(graph, sum_trusts):
    for node in graph.nodes:
        node.trust /= sum_trusts
        
        


    
        
    
def HITS_iteration(graph, n_steps, trust_included=False, trust_normalized=False, auth=-1, hub=-1, trust=-1):
    
    HITS_init(graph, auth, hub, trust)
    print_hubAuthTrust_values(graph) 
    
    for i in range(n_steps):
        HITS_one_step(graph, trust_included, trust_normalized)
    
    print_hubAuthTrust_values(graph)
    
    return copy.deepcopy(graph.nodes)
        
        
        
        

def HITS_one_step(graph, trust_included=False, trust_normalized=False, users_engaged=False):
    sum_auths = 0.
    sum_hubs = 0.
    sum_trusts = 0.
    #nodes old is required so the algorithm does not take the already updated
    #values in the for loop.
    nodes_old = copy.deepcopy(graph.nodes)
    
    if users_engaged:
        users = Graph.get_nodes_from_IDs(graph, Graph.get_user_IDs(graph))
        
        for user in users:
            
            avg_trust = Graph.get_avg_trust_of_known_nodes(user)
            rnd_document_node = Graph.get_rnd_document_node(graph)
            
            
        
    
    for node in graph.nodes:
        
        if trust_included:
            node.auth = sum(nodes_old[p].hub * nodes_old[p].trust for p in node.parents)
        else:         
            node.auth = sum(nodes_old[p].hub for p in node.parents)
        sum_auths += node.auth
        
        
        if trust_included:
            node.hub = sum(nodes_old[c].auth * nodes_old[c].trust for c in node.children)
        else:
            node.hub = sum(nodes_old[c].auth for c in node.children)
        sum_hubs += node.hub
        
        
        if trust_normalized:
            sum_parents = sum(nodes_old[p].trust * nodes_old[p].hub for p in node.parents)
            sum_children = sum(nodes_old[c].trust * nodes_old[c].auth for c in node.children)
            sum_trusts += sum_parents + sum_children
        else:
            sum_parents = sum(nodes_old[p].trust for p in node.parents)
            sum_children = sum(nodes_old[c].trust for c in node.children)
            
            if len(node.parents) > 0 or len(node.children) > 0:
                node.trust = (sum_parents + sum_children) / (len(node.parents) + len(node.children))
    
    normalize_auths(graph, sum_auths)
    normalize_hubs(graph, sum_hubs)
    
    if trust_normalized:
        normalize_trusts(graph, sum_trusts)
        

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
     
     
    



def print_hubAuth_values(graph):
    for n in graph.nodes:
        print("node", n._id, " hub = ", n.hub, "   auth =", n.auth)
    print()
    
def print_hubAuthTrust_values(graph):
    for n in graph.nodes:
        print("node", n._id, " hub = ", n.hub, "   auth =", n.auth, " trust = ", n.trust)
    print()

def print_parents_children(graph):
    for n in graph.nodes:
        print("node ", n._id, "parents: ", n.parents, ", children: ", n.children)       
    print()
    
def print_known_nodes(graph):
    for n in graph.nodes:
        if n.isUser():
            print("node ", n._id, "known nodes: ", [kn._id for kn in n.known_nodes], "avg trust: ", Graph.get_avg_trust_of_known_nodes(n))
    print()
    
def get_sorted_nodes(nodes):
    
    sorted_nodes_auth = copy.deepcopy(nodes)
    sorted_nodes_auth.sort(key=sort_by_auth)
    
    sorted_nodes_hub = copy.deepcopy(nodes)
    sorted_nodes_hub.sort(key=sort_by_hub) 

    sorted_nodes_trust = copy.deepcopy(nodes)
    sorted_nodes_trust.sort(key=sort_by_trust)
    
    return sorted_nodes_auth, sorted_nodes_hub, sorted_nodes_trust
    
    
def sort_by_trust(node):
    return node.trust

def sort_by_auth(node):
    return node.auth

def sort_by_hub(node):
    return node.hub

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
            print([node._id for node in vals[i]], "node " + value_name + " order " + param[name_id])
            
        print()
        
    print()
            
    return sorted_nodeIDs_auths, sorted_nodeIDs_hubs, sorted_nodeIDs_trusts

def plot_node_values():
    pass

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
    
    
def set_params(*args):
    params = []
    for arg in args:
        params.append(arg)
    return params
    


if __name__ == '__main__':
    
    n_steps = 20
    n_nodes = 20
    n_edges = 60
    n_users = n_nodes * 3
    n_known_nodes = 5
    #graph = Graph.create_random_weighted_directed_document_graph(n_nodes, n_edges)
    graph = Graph.load_graph("rnd_20n_60e")
    Graph.visualize(graph)
    print_parents_children(graph)
    #Graph.save_graph(graph, "rnd_20n_60e")
    
    trust_included_id = 0
    trust_normalized_id = 1
    auth_id = 2
    hub_id = 3
    trust_id = 4
    name_id = 5
    
    std_HITS_param = [False, False, 1, 1, 1, "standard hits"]
    std_HITS_rndHubAuth_param = [False, False, -1, -1, 1, "standard hits, rnd hub auth vals"]
    trust_HITS_normalized = [True, True, 1, 1, 1, "trust hits normalized"]
    rnd_trust_HITS_normalized = [True, True, 1, 1, -1, "rnd trust hits normalized"]
    rndTrust_rndHits_normalized = [True, True, -1, -1, -1, "rnd trust rnd hits normalized"]
    trust_HITS_avg = [True, False, 1, 1, 1, "trust hits avg"]
    rnd_trust_HITS_avg = [True, False, 1, 1, -1, "rnd trust hits avg"]
    
    params = set_params(std_HITS_param,
                        std_HITS_rndHubAuth_param, 
                        trust_HITS_normalized, 
                        rnd_trust_HITS_normalized,  
                        rndTrust_rndHits_normalized,  
                        trust_HITS_avg,
                        rnd_trust_HITS_avg)
    
    
    params_nodes = []
    
    sorted_nodes_auths = []
    sorted_nodes_hubs = []
    sorted_nodes_trusts = []
    
    for param in params:
        
        param_nodes = HITS_iteration(graph,
                                     n_steps, 
                                     param[trust_included_id],
                                     param[trust_normalized_id],
                                     param[auth_id],
                                     param[hub_id],
                                     param[trust_id])
        
        params_nodes.append(param_nodes)
        
        sorted_nodes_auth, sorted_nodes_hub, sorted_nodes_trust = get_sorted_nodes(param_nodes)
        sorted_nodes_auths.append(sorted_nodes_auth)
        sorted_nodes_hubs.append(sorted_nodes_hub)
        sorted_nodes_trusts.append(sorted_nodes_trust)
        
      
    
    
    
    
    
    sorted_nodeIDs_auths, sorted_nodeIDs_hubs, sorted_nodeIDs_trusts = get_sorted_nodeIDs(params, sorted_nodes_auths, sorted_nodes_hubs, sorted_nodes_trusts)    
    
    
    
    Graph.add_users(graph, 20)
    print_parents_children(graph)
    Graph.set_all_users_rnd_known_nodes(graph, n_known_nodes)
    print_known_nodes(graph)
        
        
        
    
    
    
    
    
    
    #plot_node_rankings(params, sorted_nodeIDs_auths, sorted_nodeIDs_hubs, sorted_nodeIDs_trusts)        
    #diff = mean_nodes_order_similarity([node._id for node in sorted_nodes_auths[0]], [node._id for node in sorted_nodes_auths[1]])
    #print(diff)    
        
    
    
        
        
        
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    