# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 12:29:51 2023

@author: rob
"""

import Graph_3 as G
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)



'''
notes:
    
    26.03. final step, implement semantics

    
    plotted distribution of equivalence points: it seems that the distribution is negative
    exponential. this is the kind of behavior that is wanted as a semantic value function,
    since for a search query, there will be only very few sites that make a good match, but
    exponentially more sites that do not match at all.
    
    
    
    

'''

def prime_factors(n):

    factors = []
    divisor = 2
    while divisor * divisor <= n:
        if n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        else:
            divisor += 1 if divisor == 2 else 2
    if n > 1:
        factors.append(n)
    return factors

def prime_factors_to_dict(factors, dict_to_be_extended=None):
    
    if dict_to_be_extended is None:
        factor_counts = {}
    else:
        factor_counts = dict_to_be_extended
        
    for factor in factors:
        if factor in factor_counts:
            factor_counts[factor] += 1
        else:
            factor_counts[factor] = 1
    return factor_counts

def compare_semantic_equivalence(factorsA, factorsB):
    
    equivalence_points = 0
    for factor, amount in factorsA.items():
        if factor in factorsB.keys():
            equivalence_points += min(amount, factorsB[factor]) 
            
    #TODO: check whether min or max is better
    #return equivalence_points / max(sum(factorsA.values()), sum(factorsB.values()))
    #return equivalence_points / min(sum(factorsA.values()), sum(factorsB.values()))
    return equivalence_points
            


def HITS_init(nodes, content_max):

    for n in nodes:
        n.auth = 1.
        n.hub = 1.
        G.set_content_and_private_factors(n, content_max)
        G.set_public_factors(n, )
        
        
    normalize_auths(nodes, len(nodes))
    normalize_hubs(nodes, len(nodes))
    
    
  

def set_edge_trusts(nodes, val=-1):
    for n in nodes:
        for child, weight in n.edges.items():
            n.edges.update({child : random.random()})

            
           

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
        
        
def get_root_set(nodes, size, IDs=None):
    max_index = len(nodes) - 1
    id_set = set()
    if IDs is None:
        while len(id_set) < size:
            id_set.add(random.randint(0, max_index))
        return [nodes[_id] for _id in id_set]
    else:
        return [nodes[_id] for _id in IDs]

    
    
def HITS_iteration(nodes, n_search_queries, root_set_size, n_steps,
                   enable_trust=False, 
                   auth=-1, hub=-1, trust=-1):
    
    avg_trust = G.get_avg_trust(nodes)
    print("avg trust", avg_trust)
    print_hubAuthTrust_values(nodes)
    #print_parents_children(nodes)
    
    
    for _ in range(n_search_queries):
        
        root_set_IDs = None
        root_set = get_root_set(nodes, root_set_size, root_set_IDs)
        #print([n._id for n in root_set])
        #print_hubAuthTrust_values(nodes)
    
        for _ in range(n_steps):
            HITS_one_step(nodes, root_set, enable_trust)
            #print_hubAuthTrust_values(nodes)
    
    print_hubAuthTrust_values(nodes)
    #print_parents_children(nodes)
    
    return nodes
        
        
        
        

def HITS_one_step(all_nodes, subset_nodes, enable_trust):
    
    #nodes old is required so the algorithm does not take the already updated
    #values in the for loop.
    nodes_old = copy.deepcopy(all_nodes)
            
   
    for node in subset_nodes:
        
        parents = node.parents
        children = node.children
        
        if enable_trust:   
            node.auth = sum(nodes_old[p._id].hub * nodes_old[p._id].edges.get(node._id) for p in parents)
            node.hub = sum(nodes_old[c._id].auth * nodes_old[node._id].edges.get(c._id) for c in children)
        else:
            node.auth = sum(nodes_old[p._id].hub for p in parents)
            node.hub = sum(nodes_old[c._id].auth for c in children)


    normalize_auths(subset_nodes, sum(n.auth for n in all_nodes))
    normalize_hubs(subset_nodes, sum(n.hub for n in all_nodes))
    

    
    return nodes

        

    
def get_sorted_nodes(nodes):
    
    sorted_nodes_auth = copy.deepcopy(nodes)
    sorted_nodes_auth.sort(key=sort_by_auth)
    
    sorted_nodes_hub = copy.deepcopy(nodes)
    sorted_nodes_hub.sort(key=sort_by_hub) 
    
    return sorted_nodes_auth, sorted_nodes_hub


def sort_by_auth(node):
    return node.auth

def sort_by_hub(node):
    return node.hub

def get_sorted_nodeIDs(params, sorted_nodes_auths, sorted_nodes_hubs):
    
    sorted_nodeIDs_auths = []
    sorted_nodeIDs_hubs = []

    
    for i in range(2):
        if i == 0:
            value_name = "AUTH: "
            vals = sorted_nodes_auths
            sorted_nodeIDs = sorted_nodeIDs_auths
            
        else:
            value_name = "HUB: "
            vals = sorted_nodes_hubs
            sorted_nodeIDs = sorted_nodeIDs_hubs

                      
        for i, param in enumerate(params):
            sorted_nodeIDs.append([node._id for node in vals[i]])
            print([node._id for node in vals[i]], value_name + " " + param[name_id])
            
        print()
        
    print()
            
    return sorted_nodeIDs_auths, sorted_nodeIDs_hubs



    
    



'''_______________________________UTILS_____________________________________'''  

def set_params(*args):
    return [arg for arg in args] 
        

def print_hubAuthTrust_values(nodes):
    for n in nodes:
        print(n._id, " hub = ", n.hub, "   auth =", n.auth)
    print()

def print_parents_children(nodes):
    for n in nodes:
        print("node ", n._id, "parents: ", [p._id for p in n.parents], ", children: ", [c._id for c in n.children], "edges: ", [(e, v) for e, v in n.edges.items()])   
        
    print()
    
def plot_equivalence_points_distribution():

    b = 9
    boxes = np.zeros(b)
    n = 2000

    for _ in range(n):
    
        idA = random.randint(2, 500000000)
        idB = random.randint(2, 500)    
        
        factorsA = prime_factors(idA)
        factorsB = prime_factors(idB)

        dA = prime_factors_to_dict(factorsA)
        dB = prime_factors_to_dict(factorsB)

        c = compare_semantic_equivalence(dA, dB)
        
        boxes[c] += 1
        


    plt.bar(range(b), boxes)
    plt.show()
    
    
    
'''_______________________________UTILS_____________________________________'''
    


if __name__ == '__main__':
    
    n_steps = 5
    n_nodes = 20
    n_edges = 60
    n_users = n_nodes
    n_search_queries = 20
    root_set_size = 5
    
    n_search_queries_id = 0
    root_set_size_id = 1
    enable_trust_id = 2
    users_engaged_id = 3
    name_id = 4
    
    
    
    #params corresponding to the above definitions
    
    std_hits = [1, n_nodes, False, False, "std HITS, no users"]

    
    
    
    #base_nodes = G.create_random_weighted_directed_document_nodes(n_nodes, n_edges)
    base_nodes = G.load_graph("rnd_20n_60e_3")
    G.visualize(base_nodes)
    print_parents_children(base_nodes)
    #G.save_graph(base_nodes, "rnd_20n_60e_3")
    
    params = set_params(std_hits)
    
    #params = [params[-1]]
    
    
    #params = [params[0], params[-1]]
    
    node_copies = [copy.deepcopy(base_nodes) for nodes in range(len(params))]
    
    
    sorted_nodes_auths = []
    sorted_nodes_hubs = []

    
    for nodes, param in zip(node_copies, params):
        
        print("\n\n", param[name_id], "\n")
        #print_parents_children(nodes)
        HITS_init(nodes)
        HITS_iteration(nodes,
                       param[n_search_queries_id],
                       param[root_set_size_id],
                       n_steps, 
                       param[enable_trust_id])
        
        
        #print_parents_children(nodes)

        sorted_nodes_auth, sorted_nodes_hub = get_sorted_nodes(nodes)
        sorted_nodes_auths.append(sorted_nodes_auth)
        sorted_nodes_hubs.append(sorted_nodes_hub)

    
    sorted_nodeIDs_auths, sorted_nodeIDs_hubs = get_sorted_nodeIDs(params, sorted_nodes_auths, sorted_nodes_hubs)    
     

        
    
    
       
        
        
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    