# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:48:17 2023

@author: rob
"""

import random
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt



class Node:
    
    def __init__(self, _id, _type):
       self._id = _id
       self._type = _type
       self.children = []
       self.parents = []
       self.edges = dict()
       self.auth = 0.
       self.hub = 0.
       self.trust = 0.
       self.known_nodes = set()
       
    def __hash__(self):
        return hash(self._id)
    
    def __eq__(self, other):
        return self._id == other._id  
    
    def __str__(self):
        return str(self._id)

    def isDocument(self):
        return self._type == 'document'
     
    def isUser(self):
        return self._type == 'user'
    

    
        
def EdgeTrustInit():
    return 1       
     
def NodeType_Document():
    return 'document'

def NodeType_User():
    return 'user'
        
'''def get_node(nodes, _id):
    return nodes[_id]'''

'''def get_node(nodes, _id):
    for n in nodes:
        if n._id == _id:
            return n
    return None'''
    
        
def get_edges(nodes):
    edges = []
    for n in nodes:
        for child in n.children:
            edges.append((n, child))
    return edges
        
        
    
def get_n_documents(nodes):
    n_documents = 0
    for n in nodes:
        if n._type == NodeType_Document():
            n_documents += 1
    return n_documents

def get_n_users(nodes):
    n_users = 0
    for n in nodes:
        if n._type == NodeType_User():
            n_users += 1
    return n_users



def get_document_IDs(nodes):
    return set([n._id for n in nodes if n.isDocument()])
    
def get_user_IDs(nodes):
    return set([n._id for n in nodes if n.isUser()])

def get_users(nodes):
    return get_nodes_from_IDs(nodes, get_user_IDs(nodes))

def get_documents(nodes):
    return get_nodes_from_IDs(nodes, get_document_IDs(nodes))
    
def add_users(nodes, n_users):
    first_id = len(nodes)
    last_id = first_id + n_users
    users = [Node(i, NodeType_User()) for i in range(first_id, last_id)] 
    nodes.extend(users)     
    
def get_rnd_known_nodes(nodes, n_known_nodes):
    node_ids = random.sample(get_document_IDs(nodes), n_known_nodes)
    return set([nodes[id] for id in node_ids])

def set_all_users_rnd_known_nodes(nodes, n_known_nodes):
    for n in nodes:
        if n.isUser():
            n.known_nodes = get_rnd_known_nodes(nodes, n_known_nodes)
            
def get_nodes_from_IDs(nodes, IDs):
    return [nodes[_id] for _id in IDs]

            
def get_avg_trust_of_known_nodes(node):
    return np.mean([kn.trust for kn in node.known_nodes])

def get_rnd_document_node(nodes):
    max_index = len(nodes) - 1
    document_node = nodes[random.randint(0, max_index)]
    while document_node.isUser():
        document_node = nodes[random.randint(0, max_index)]
    return document_node
        
def create_random_weighted_directed_document_nodes(n_nodes, n_edges):
    
    nodes = [Node(i, NodeType_Document()) for i in range(n_nodes)]
    _edges = set()
    #auxiliary set that helps the while loop create exactly n_edges
    #_edges is not used beyond this function
    max_index = n_nodes - 1
    while len(_edges) < n_edges:
        parent_id = random.randint(0, max_index)
        child_id = random.randint(0, max_index)
        if (parent_id, child_id) not in _edges:
            parent = nodes[parent_id]         
            child = nodes[child_id]
            parent.children.append(child)
            child.parents.append(parent)
            parent.edges.update({child_id : random.random()})
            _edges.add((parent_id, child_id))
    
    return nodes


'''________________________UTILS_________________________________________'''

def convert_nodes_to_networkx(nodes):
    nxGraph = nx.DiGraph()
    edges = get_edges(nodes)
    nx_edges = []
    for parent, child in edges:
        nx_edges.append((parent._id, child._id, parent.edges.get(child._id)))
    nxGraph.add_weighted_edges_from(nx_edges)
    return nxGraph

def visualize(nodes):
    plt.rcParams['figure.dpi'] = 600
    nxGraph = convert_nodes_to_networkx(nodes)
    nx.draw_networkx(nxGraph, pos=nx.spring_layout(nxGraph), with_labels=True)
    plt.title("Webgraph with " + str(nxGraph.number_of_nodes()) + " Nodes and " + str(nxGraph.number_of_edges()) + " Edges")
    plt.axis('off')
    plt.show()
    
def save_graph(graph, filename):
    with open("../data/graphs/" + filename + ".pkl", "wb") as f:
        pickle.dump(graph, f)
        
def load_graph(filename):
    with open("../data/graphs/" + filename + ".pkl", "rb") as f:
        graph = pickle.load(f)
    return graph

'''________________________UTILS_________________________________________'''






'''__________________________OUTDATED_________________________________'''
'''







'''
'''__________________________OUTDATED_________________________________'''
    
    

















        
    
    
        
    
    

