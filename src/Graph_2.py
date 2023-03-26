# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:38:02 2023

@author: rob
"""


import random
import networkx as nx
import pickle
import numpy as np


class User:
    
    def __init__(self, _id):
        self._id = _id
        self.children = []
        self.edges = dict()
        
        
    def __hash__(self):
        return hash(self._id)
    
    def __eq__(self, other):
        return self._id == other._id  
    
    def __str__(self):
        return str(self._id)
    
    def get_avg_trust(self):
        avg_trust = 0.
        for trust_val in self.edges.values():
            avg_trust += 1
        return avg_trust
    
    def adjust_children(self, nodes):
        avg_auth = (sum(n.auth for n in self.children) + sum(n.auth for n in nodes)) / (len(self.children) + len(nodes))
        new_children = []
        new_children_IDs = set()
        both_node_lists = self.children + nodes
        for n in both_node_lists:
            if n.auth > avg_auth:
                if n._id not in new_children_IDs:
                    new_children.append(n)
                    new_children_IDs.add(n._id)
        self.children = new_children
    
def select_rnd_user(users):
    max_idx = len(users) - 1
    return users[random.randint(0, max_idx)]



class Node:
    
    def __init__(self, _id):
       self._id = _id
       self.children = []
       self.parents = []
       self.edges = dict()
       self.auth = 1.
       self.hub = 1.

       
    def __hash__(self):
        return hash(self._id)
    
    def __eq__(self, other):
        return self._id == other._id  
    
    def __str__(self):
        return str(self._id)

    
def get_users_avg_trust(users):
    cusum = 0.
    count = 0
    for u in users:
        for c in u.children:
            for p in c.parents:
                cusum += p.edges.get(c._id)
            count += len(c.parents)
    return cusum / count

def get_avg_trust(nodes):
    cusum = 0.
    count = 0
    for n in nodes:
        cusum += sum(weight for weight in n.edges.values())
        count += len(n.edges)
    return cusum / count
        
def get_edges(nodes):
    edges = []
    for n in nodes:
        for child in n.children:
            edges.append((n, child))
    return edges
        
            
def get_nodes_from_IDs(nodes, IDs):
    return [nodes[_id] for _id in IDs]


def create_users(n_users):
    return [User(i) for i in range(n_users)]
        
def create_random_weighted_directed_document_nodes(n_nodes, n_edges):
    
    nodes = [Node(i) for i in range(n_nodes)]
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
    nxGraph = convert_nodes_to_networkx(nodes)
    nx.draw_networkx(nxGraph, with_labels=True)
    
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
    
    

















        
    
    
        
    
    

