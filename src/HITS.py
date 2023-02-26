# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:06:25 2023

@author: rob
"""

import Graph




def HITS_one_step(graph):
    auth_total = 0.
    hub_total = 0.
    for node in graph.nodes:
        node.auth = sum(graph.nodes[p].hub for p in node.parents)
        auth_total += node.auth
        
    for node in graph.nodes:
        node.hub = sum(graph.nodes[c].auth for c in node.children)
        hub_total += node.hub
        
    for node in graph.nodes:
        node.auth /= auth_total
        node.hub /= hub_total
    
    
    
        
        
    
    


def print_hubAuth_values(graph):
    for n in graph.nodes:
        print("node", n._id, " hub = ", n.hub, "   auth =", n.auth)

def print_parents_children(graph):
    for n in graph.nodes:
        print("node ", n._id, "parents: ", n.parents, ", children: ", n.children)




if __name__ == '__main__':
    
    steps = 5
    graph = Graph.create_random_weighted_directed_document_graph(30, 120)
    graph.visualize()
    print_hubAuth_values(graph)
    print_parents_children(graph)
    
    
    for i in range(steps):
        HITS_one_step(graph)
        print_hubAuth_values(graph)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    