# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:06:25 2023

@author: rob
"""

import Graph
import random
import copy


'''
insights:
    
    fixed initial trust values do not make sense,
    because trust just becomes a measure of overall connectedness.
    
    this also applies to the hub and authorities values. if uniformly
    initialized at 1.0, they solely become a measure of connectedness...
    
    the deciding factor is indeed the prior requried for the initialization
    
    ==> the prior is actually not the deciding factor. see node trust order
    



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
        node.setAuth(auth)
    normalize_auths(graph, sum(auths))
       
def normalize_auths(graph, sum_auths):
    for node in graph.nodes:
        node.auth /= sum_auths
        
        
def set_all_hubs(graph, hubs):
    for node, hub in zip(graph.nodes, hubs):
        node.setHub(hub)
    normalize_hubs(graph, sum(hubs))  
        
def normalize_hubs(graph, sum_hubs):
    for node in graph.nodes:
        node.hub /= sum_hubs
        
        
def set_all_trusts(graph, trusts):
    for node, trust in zip(graph.nodes, trusts):
        node.setTrust(trust)
    normalize_trusts(graph, sum(trusts))
        
def normalize_trusts(graph, sum_trusts):
    for node in graph.nodes:
        node.trust /= sum_trusts
    

        
    
        
        
        
        
        

def HITS_one_step(graph, weighted_trust=False, trust_adjustment=False):
    sum_auths = 0.
    sum_hubs = 0.
    sum_trusts = 0.
    #nodes old is required so the algorithm does not take the already updated
    #values in the for loop.
    nodes_old = copy.deepcopy(graph.nodes)
    for node in graph.nodes:
        if weighted_trust:
            node.auth = sum(nodes_old[p].hub * nodes_old[p].trust for p in node.parents)
        else:         
            node.auth = sum(nodes_old[p].hub for p in node.parents)
        sum_auths += node.auth
        
    for node in graph.nodes:
        if weighted_trust:
            node.hub = sum(nodes_old[c].auth * nodes_old[c].trust for c in node.children)
        else:
            node.hub = sum(nodes_old[c].auth for c in node.children)
        sum_hubs += node.hub
        
    for node in graph.nodes:
        if trust_adjustment:
            sum_parents = sum(nodes_old[p].trust * nodes_old[p].hub for p in node.parents)
            sum_children = sum(nodes_old[c].trust * nodes_old[c].auth for c in node.children)
        else:
            sum_parents = sum(nodes_old[p].trust for p in node.parents)
            sum_children = sum(nodes_old[c].trust for c in node.children)
        node.trust = sum_parents + sum_children
        sum_trusts += sum_parents + sum_children
    
    normalize_auths(graph, sum_auths)
    normalize_hubs(graph, sum_hubs) 
    normalize_trusts(graph, sum_trusts)
        

        
    



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
    
    
def sort_by_trust(node):
    return node.trust

def sort_by_auth(node):
    return node.auth

def sort_by_hub(node):
    return node.hub


if __name__ == '__main__':
    
    steps = 50
    graph = Graph.create_random_weighted_directed_document_graph(75, 200)
    #graph.visualize()
    
    HITS_init(graph, -1, -1, -1)
    print_hubAuthTrust_values(graph)
    print_parents_children(graph)
    
    
    for i in range(steps):
        HITS_one_step(graph)
    print_hubAuthTrust_values(graph)
    sorted_nodes1 = copy.deepcopy(graph.nodes)
    sorted_nodes1.sort(key=sort_by_trust)
    
    sorted_nodes1_auth = copy.deepcopy(graph.nodes)
    sorted_nodes1_auth.sort(key=sort_by_auth)
    
    sorted_nodes1_hub = copy.deepcopy(graph.nodes)
    sorted_nodes1_hub.sort(key=sort_by_hub)
    
        
        
    HITS_init(graph, 1.0, 1.0, 1.0) 
    print_hubAuthTrust_values(graph)
    print_parents_children(graph)
    
    
    for i in range(steps):
        HITS_one_step(graph)
    print_hubAuthTrust_values(graph)
        
    sorted_nodes2 = copy.deepcopy(graph.nodes)
    sorted_nodes2.sort(key=sort_by_trust)
    
    sorted_nodes2_auth = copy.deepcopy(graph.nodes)
    sorted_nodes2_auth.sort(key=sort_by_auth)
    
    sorted_nodes2_hub = copy.deepcopy(graph.nodes)
    sorted_nodes2_hub.sort(key=sort_by_hub)
    
    
    HITS_init(graph, 1.0, 1.0, 1.0) 
    print_hubAuthTrust_values(graph)
    print_parents_children(graph)
    
    
    for i in range(steps):
        HITS_one_step(graph, True)
    print_hubAuthTrust_values(graph)
        
    sorted_nodes3 = copy.deepcopy(graph.nodes)
    sorted_nodes3.sort(key=sort_by_trust)
    
    sorted_nodes3_auth = copy.deepcopy(graph.nodes)
    sorted_nodes3_auth.sort(key=sort_by_auth)
    
    sorted_nodes3_hub = copy.deepcopy(graph.nodes)
    sorted_nodes3_hub.sort(key=sort_by_hub)
    
    
    HITS_init(graph, 1.0, 1.0, 1.0) 
    print_hubAuthTrust_values(graph)
    print_parents_children(graph)
    
    
    for i in range(steps):
        HITS_one_step(graph, True, True)
    print_hubAuthTrust_values(graph)
        
    sorted_nodes4 = copy.deepcopy(graph.nodes)
    sorted_nodes4.sort(key=sort_by_trust)
    
    sorted_nodes4_auth = copy.deepcopy(graph.nodes)
    sorted_nodes4_auth.sort(key=sort_by_auth)
    
    sorted_nodes4_hub = copy.deepcopy(graph.nodes)
    sorted_nodes4_hub.sort(key=sort_by_hub)
    
    
    HITS_init(graph, -1, -1, -1) 
    print_hubAuthTrust_values(graph)
    print_parents_children(graph)
    
    
    for i in range(steps):
        HITS_one_step(graph, True, True)
    print_hubAuthTrust_values(graph)
        
    sorted_nodes5 = copy.deepcopy(graph.nodes)
    sorted_nodes5.sort(key=sort_by_trust)
    
    sorted_nodes5_auth = copy.deepcopy(graph.nodes)
    sorted_nodes5_auth.sort(key=sort_by_auth)
    
    sorted_nodes5_hub = copy.deepcopy(graph.nodes)
    sorted_nodes5_hub.sort(key=sort_by_hub)
    
    
    print("node trust order 1", [node._id for node in sorted_nodes1])
    print("node trust order 2", [node._id for node in sorted_nodes2])
    print("node trust order 3", [node._id for node in sorted_nodes3])
    print("node trust order 4", [node._id for node in sorted_nodes4])
    print("node trust order 5", [node._id for node in sorted_nodes5])
    
    
    print("node auth order 1 ", [node._id for node in sorted_nodes1_auth])
    print("node auth order 2 ", [node._id for node in sorted_nodes2_auth])
    print("node auth order 3 ", [node._id for node in sorted_nodes3_auth])
    print("node auth order 4 ", [node._id for node in sorted_nodes4_auth])
    print("node auth order 5 ", [node._id for node in sorted_nodes5_auth])
    
    print("node hub order 1  ", [node._id for node in sorted_nodes1_hub])
    print("node hub order 2  ", [node._id for node in sorted_nodes2_hub])
    print("node hub order 3  ", [node._id for node in sorted_nodes3_hub])
    print("node hub order 4  ", [node._id for node in sorted_nodes4_hub])
    print("node hub order 5  ", [node._id for node in sorted_nodes5_hub])
    
    
    
    
    
        
        
        
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    