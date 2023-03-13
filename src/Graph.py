# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:48:17 2023

@author: rob
"""

import random
import networkx as nx
import pickle
import numpy as np



class Node:
    
    def __init__(self, _id, _type):
       self._id = _id
       self._type = _type
       self.children = set()
       self.parents = set()
       self.auth = 0.
       self.hub = 0.
       self.trust = 0.
       self.known_nodes = set()
       
    def __hash__(self):
        return hash(self._id)
    
    def __eq__(self, other):
        return self._id == other._id  

    def isDocument(self):
        return self._type == 'document'
     
    def isUser(self):
        return self._type == 'user'
    
        
        
     
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
    
def get_n_documents(graph):
    n_documents = 0
    for n in graph.nodes:
        if n._type == NodeType_Document():
            n_documents += 1
    return n_documents

def get_n_users(graph):
    n_users = 0
    for n in graph.nodes:
        if n._type == NodeType_User():
            n_users += 1
    return n_users

def visualize(graph):
    nxGraph = convert_digraph_to_networkx(graph)
    nx.draw_networkx(nxGraph, with_labels=True)
    
def save_graph(graph, filename):
    with open("../data/graphs/" + filename + ".pkl", "wb") as f:
        pickle.dump(graph, f)
        
def load_graph(filename):
    with open("../data/graphs/" + filename + ".pkl", "rb") as f:
        graph = pickle.load(f)
    return graph

def get_document_IDs(graph):
    return set([n._id for n in graph.nodes if n.isDocument()])
    
def get_user_IDs(graph):
    return set([n._id for n in graph.nodes if n.isUser()])
    
def add_users(graph, n_users):
    first_id = len(graph.nodes)
    last_id = first_id + n_users
    users = [Node(i, NodeType_User()) for i in range(first_id, last_id)] 
    graph.nodes.extend(users)     
    
def get_rnd_known_nodes(graph, n_known_nodes):
    node_ids = random.sample(get_document_IDs(graph), n_known_nodes)
    return set([graph.nodes[id] for id in node_ids])

def set_all_users_rnd_known_nodes(graph, n_known_nodes):
    for n in graph.nodes:
        if n.isUser():
            n.known_nodes = get_rnd_known_nodes(graph, n_known_nodes)
            
def get_nodes_from_IDs(graph, IDs):
    return [graph.nodes[ID] for ID in IDs]
            
def get_avg_trust_of_known_nodes(node):
    return np.mean([kn.trust for kn in node.known_nodes])

def get_rnd_document_node(graph):
    document_node = graph.nodes[random.randint(0, len(graph.nodes))]
    while document_node.isUser():
        document_node = graph.nodes[random.randint(0, len(graph.nodes))]
    return document_node
        
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
    
    

















        
    
    
        
    
    

