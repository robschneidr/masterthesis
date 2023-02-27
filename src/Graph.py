# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:48:17 2023

@author: rob
"""

import random
import networkx as nx



class Node:
    
    def __init__(self, _id, _type):
       self._id = _id
       self._type = _type
       self.children = set()
       self.parents = set()
       self.auth = 0.
       self.hub = 0.
       self.trust = 0.
       
    def __hash__(self):
        return hash(self._id)
    
    def __eq__(self, other):
        return self._id == other._id  

    def isDocument(self):
        return self._type == 'document'
     
    def isUser(self):
        return self._type == 'user'
    
    def setAuth(self, auth):
        self.auth = auth
        
    def setHub(self, hub):
        self.hub = hub
    
    def setTrust(self, trust):
        self.trust = trust
        
        
     
def NodeType_Document():
    return 'document'

def NodeType_User():
    return 'user'


class DirectedWeightedEdge:
    
    def __init__(self, parent, child, weight):
        self.parent = parent
        self.child = child
        self.weight = weight
        
    def __hash__(self):
        return hash((self.parent, self.child))
    
    def __eq__(self, other):
        return self.parent == other.parent and self.child == other.child
        
        
    def data_to_tuple(self):
        return (self.parent, self.child, self.weight)



class Graph:
    
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        set_adjacency_sets(self.nodes, self.edges)
    
    def visualize(self):
        nxGraph = convert_digraph_to_networkx(self)
        nx.draw_networkx(nxGraph, with_labels=True)
        
        
        

def create_random_weighted_directed_document_graph(n_nodes, n_edges):
    
    nodes = [Node(i, NodeType_Document()) for i in range(n_nodes)]
    edges = {}
    while len(edges) < n_edges:
        parent = random.randint(0, n_nodes - 1)
        child = random.randint(0, n_nodes - 1)
        edges.update({(parent, child) : DirectedWeightedEdge(parent, child, random.random())}) 
    set_adjacency_sets(nodes, edges)
    return Graph(nodes, edges)


def set_adjacency_sets(nodes, edges):
    for e in edges.values():
        nodes[e.parent].children.add(e.child)
        nodes[e.child].parents.add(e.parent)

def convert_digraph_to_networkx(digraph):
    nxGraph = nx.DiGraph()
    nxGraph.add_weighted_edges_from([e.data_to_tuple() for e in digraph.edges.values()])
    return nxGraph
    
    

















        
    
    
        
    
    

