# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 09:59:55 2023

@author: rob
"""


import random
import networkx as nx
import pickle
import numpy as np
import HITS_4 as HITS
import copy



class Node:
    
    def __init__(self, _id):
       self._id = _id
       self.children = []
       self.parents = []
       self.edges = dict()
       self.auth = 1.
       self.hub = 1.
       self.content = 0
       self.private_factors = dict()
       self.public_factors = dict()
       self.false_factor_probability = HITS.FALSE_FACTOR_FLOOR

       
    def __hash__(self):
        return hash(self._id)
    
    def __eq__(self, other):
        return self._id == other._id  
    
    def __str__(self):
        return str(self._id)


def init_content(node, content_max):
    node.content = random.randint(HITS.RANDINT_PRIMES_FLOOR, content_max)
    node.private_factors = HITS.prime_factors_to_dict(HITS.prime_factors(node.content))
    node.public_factors = copy.deepcopy(node.private_factors)

    
def set_all_edge_weights(nodes, weight):
    for n in nodes:
        for child in n.edges.keys():
            n.edges.update({child : weight})
            
def get_edges(nodes):
    edges = []
    for n in nodes:
        for child in n.children:
            edges.append((n, child))
    return edges

def get_edge_ids(nodes):
    edges = set()
    for n in nodes:
        for child in n.children:
            edges.add((n._id, child._id))
    return edges

 
    
def get_avg_false_factor_probability(nodes):
    return sum(n.false_factor_probability for n in nodes) / len(nodes)

def get_avg_trust(nodes):
    cusum = 0.
    count = 0
    for n in nodes:
        cusum += sum(weight for weight in n.edges.values())
        count += len(n.edges)
    return cusum / count

def get_n_parentless(nodes):
    return sum([1 for n in nodes if len(n.parents) == 0])
        
def get_n_edges(nodes):
    return sum([len(n.edges) for n in nodes])
        
            
def get_nodes_from_IDs(nodes, IDs):
    return [nodes[_id] for _id in IDs]

def get_node_IDs(nodes):
    return set([n._id for n in nodes])

def remove_untrustworthy_edges(nodes, threshold):
    n_removed_edges = 0
    for n in nodes:
        edges_to_be_removed = set()
        for child, trust in n.edges.items():
                  
            if trust < threshold:
                #print("node", n._id, "child", child, "trustworthiness", trust)
                #print("children before: ", [m._id for m in n.children])
                #print("parents of child before: ", [m._id for m in nodes[child].parents])
                #print("edges before: ", n.edges.items())
                n_removed_edges += 1
                del nodes[child].parents[nodes[child].parents.index(n)]
                del n.children[n.children.index(nodes[child])]
                edges_to_be_removed.add(child)
                #print("children after: ", [m._id for m in n.children])
                #print("parents of child after: ", [m._id for m in nodes[child].parents])
                
        for e in edges_to_be_removed:
            del n.edges[e]
        #if edges_to_be_removed:
            #print("edges after: ", n.edges.items())
            #print()
    return n_removed_edges
        

def replace_parentless_nodes(nodes, content_max):
    n_lost_edges = 0
    for n in nodes:
        if len(n.parents) == 0:
            #delete the parent pointers of the children
            n_lost_edges += len(n.children)
            for c in n.children:
                del c.parents[c.parents.index(n)] 
            nodes[n._id] = Node(n._id)
            init_content(nodes[n._id], content_max)

    return n_lost_edges

def replace_edges(nodes, n_lost_edges, edge_init):
    _edges = get_edge_ids(nodes)
    previous_size = len(_edges)
    max_index = len(nodes) - 1
    while len(_edges) < previous_size + n_lost_edges:
        parent_id = random.randint(0, max_index)
        parent = nodes[parent_id]
        ranking = HITS.get_ranking([(m._id, m.public_factors) for m in nodes if m._id != parent._id], parent.private_factors)
        child_id = nodes[ranking.pop()]._id
        if (parent_id, child_id) not in _edges:
            parent = nodes[parent_id]         
            child = nodes[child_id]
            parent.children.append(child)
            child.parents.append(parent)
            parent.edges.update({child_id : edge_init})
            _edges.add((parent_id, child_id))


        


def create_semantic_connection_nodes(n_nodes, n_edges, content_max):
    nodes = [Node(i) for i in range(n_nodes)]
    for n in nodes:
        init_content(n, content_max)
    rankings = []
    for n in nodes:
        ranking = HITS.get_ranking([(m._id, m.public_factors) for m in nodes if n._id != m._id], n.private_factors)
        
        '''
        see 17.04. notes in HITS_4
        ranking.reverse()
        '''

        rankings.append(ranking)
    _edges = set()
    max_index = n_nodes - 1
    while len(_edges) < n_edges:
        parent_id = random.randint(0, max_index)
        child_id = nodes[rankings[parent_id].pop()]._id
        if (parent_id, child_id) not in _edges:
            parent = nodes[parent_id]         
            child = nodes[child_id]
            parent.children.append(child)
            child.parents.append(parent)
            parent.edges.update({child_id : random.random()})
            _edges.add((parent_id, child_id))
            
    return nodes
        
            

        
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
    
    

















        
    
    
        
    
    

