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
import math

np.set_printoptions(precision=3, suppress=True)
RANDINT_PRIMES_FLOOR = 2



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
    
    '''
    
    TODO: to get a more detailed model of semantic equivalence, there are two additional measures:
      1.) weigh the factors by their probability of occurring: eg. the integer 2 will occur on every
          second number, so it should be weighted 1 - (1/2), for 3: 1 - (1/3) etc
          
      2.) penalize big remaining primes. example: search query [2, 2, 5] should best describe an object
          that consists of lots of 2 and 5 factors, like [2, 2, 2, 2, 5, 5, 5, 5, 5]. but if the object,
          although the overall size is the same, like [2, 2, 5, 1820387], has large primes left,
          the similarity should be penalized.
          
          important: check that the scales of the 3 components (points, factorprob, bigprime) in the calculation
          are the same, ie -> all meta-operations (like mult is a meta operation of add and exp is meta of mult)
          need to be inversed to achieve the same scale
          
          observation: it is ok to use the counting function to generate equivalence points, as long as these points
          are penalized with the value of factors and the prime remainders.
    
    '''
    equivalence_points = 0
    sumA = sum(factorsA.keys())
    sumB = sum(factorsB.keys())
    
    for factor, amount in factorsA.items():
        if factor in factorsB:
            penalty_for_often_occurring_factors = (1 - 1 / factor)        
            equivalence_points += min(amount, factorsB[factor]) * penalty_for_often_occurring_factors

    _scaler = math.log(max(sumA, sumB))
    penalty_for_large_prime_factors = (1 - (abs(sumA - sumB) / max(sumA, sumB))) ** (1 / _scaler)

    return equivalence_points * penalty_for_large_prime_factors

def check_public_factors(node, public_factors, content):
    
    #although not entirely correct, this saves computation time..
    #real_factors = prime_factors_to_dict(prime_factors(content))
    real_factors = node.private_factors
    
    #TODO find a measure for how much computing it needs
    n_false_factors = 0
    for public_factor, amount in public_factors.items():       
        if public_factor in real_factors:
            n_false_factors += abs(amount - real_factors[public_factor]) / public_factor
        else:
            n_false_factors += amount / public_factor
    return n_false_factors

def get_private_ranking(nodes, query_factors):
    
    equivalence_points = dict()
    for n in nodes:
        equivalence_points[n._id] = compare_semantic_equivalence(query_factors, n.private_factors)
        print(n._id, equivalence_points[n._id])
    equivalence_points = sorted(equivalence_points, key=equivalence_points.get)
    return equivalence_points
    
        
            
    
            


def HITS_init(nodes, content_max):

    for n in nodes:
        n.auth = 1.
        n.hub = 1.
        G.set_content_and_private_factors(n, content_max)
        G.set_public_factors(n)
        
        
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
        
        
def get_root_set(nodes, size, content_max, query_factors_scaling):
    max_index = len(nodes) - 1
    id_set = set()
    if size > 1:
        while len(id_set) < size:
            id_set.add(random.randint(0, max_index))
    else:
        query_factors_content = random.randint(RANDINT_PRIMES_FLOOR, int(content_max ** (1 / query_factors_scaling)))
        query_factors = list(prime_factors_to_dict(prime_factors(query_factors_content)).keys())
        #print(int(content_max ** (1 / query_factors_scaling)), query_factors_scaling, content_max, query_factors_content, query_factors)
          
        for n in nodes:
            for f in query_factors:
                if f in n.public_factors:
                    id_set.add(n._id)
                    
    return [nodes[_id] for _id in id_set]


def set_trust(nodes, willingness_to_compute, learning_rate):
    
    #TODO: penalize low value false factors, because they are most likely to be 
    #added to the root set
    
    for n in nodes:
        if random.random() < willingness_to_compute:
            #print("node ", n._id, "is willing to compute:")
            for c in n.children:
               n_false_factors = check_public_factors(c, c.public_factors, c.content)
               n.edges[c._id] = max(0, min(1, n.edges[c._id] + learning_rate * (1 - n_false_factors)))
               '''print("checked child ", c._id, "content: ", c.content, "false factors: ", n_false_factors) 
               print("public factors: ", c.public_factors)
               print("private factors: ", c.private_factors)
               print()'''
               
               

def set_false_factors(nodes, false_factor_probability):
    
    for n in nodes:
        if random.random() < false_factor_probability:
            false_factors = prime_factors(random.randint(RANDINT_PRIMES_FLOOR, RANDINT_PRIMES_FLOOR + int(random.expovariate(1 / math.log(n.content)))))
            for false_factor in false_factors:
                if false_factor in n.public_factors:
                    n.public_factors[false_factor] += 1
                else:
                    n.public_factors[false_factor] = 1
                
                

        
       

    
    
def HITS_iteration(nodes, n_search_queries, root_set_size=-1, n_steps=5,
                   enable_trust=False, content_max=1000, query_factors_scaling=1,
                   false_factor_probability=0.1, willingness_to_compute=0.5, learning_rate=0.001):
    
    '''avg_trust = G.get_avg_trust(nodes)
    print("avg trust", avg_trust)
    print_hubAuthTrust_values(nodes)
    #print_parents_children(nodes)'''
    
    
    
    for _ in range(n_search_queries):
        
        set_false_factors(nodes, false_factor_probability)
        set_trust(nodes, willingness_to_compute, learning_rate)
        print_parents_children(nodes)
        
        root_set = get_root_set(nodes, root_set_size, content_max, query_factors_scaling)
        #print_hubAuthTrust_values(nodes)
        print([n._id for n in root_set])
        
        
    
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
    
        idA = random.randint(RANDINT_PRIMES_FLOOR, 500000000)
        idB = random.randint(RANDINT_PRIMES_FLOOR, 500)    
        
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
    
    n_steps = 20
    n_nodes = 20
    n_edges = 60
    n_users = n_nodes
    n_search_queries = 50
    root_set_size = 5
    std_learning_rate = 0.01
    
    n_search_queries_id = 0
    root_set_size_id = 1
    enable_trust_id = 2
    content_max_id = 3
    query_factors_scaling_id = 4
    false_factor_probability_id = 5
    willingness_to_compute_id = 6
    learning_rate_id = 7
    name_id = 8
    
    #params corresponding to the above definitions
    
    std_hits = [1, n_nodes, False, 10**10, 6, 0.1, 0.5, 0., "std HITS"]
    semantic_std_hits = [n_search_queries, -1, False, 10**10, 4, 0.1, 0.5, std_learning_rate, "semantic std HITS"]
    semantic_trust_hits = [n_search_queries, -1, True, 10**10, 4, 0.1, 0.5, std_learning_rate, "semantic trust HITS"]
    

    
    
    
    #base_nodes = G.create_random_weighted_directed_document_nodes(n_nodes, n_edges)
    base_nodes = G.load_graph("rnd_20n_60e_3")
    G.visualize(base_nodes)
    print_parents_children(base_nodes)
    #G.save_graph(base_nodes, "rnd_20n_60e_3")
    
    params = set_params(std_hits,
                        semantic_std_hits,
                        semantic_trust_hits)
    
    #params = [params[-1]]
    
    
    #params = [params[0], params[-1]]
    
    node_copies = [copy.deepcopy(base_nodes) for nodes in range(len(params))]
    
    
    sorted_nodes_auths = []
    sorted_nodes_hubs = []

    
    for nodes, param in zip(node_copies, params):
        
        print("\n\n", param[name_id], "\n")
        #print_parents_children(nodes)
        HITS_init(nodes, param[content_max_id])
        HITS_iteration(nodes,
                       param[n_search_queries_id],
                       param[root_set_size_id],
                       n_steps, 
                       param[enable_trust_id],
                       param[content_max_id],
                       param[query_factors_scaling_id],
                       param[false_factor_probability_id],
                       param[willingness_to_compute_id])
        
        
        #print_parents_children(nodes)

        sorted_nodes_auth, sorted_nodes_hub = get_sorted_nodes(nodes)
        sorted_nodes_auths.append(sorted_nodes_auth)
        sorted_nodes_hubs.append(sorted_nodes_hub)

    
    sorted_nodeIDs_auths, sorted_nodeIDs_hubs = get_sorted_nodeIDs(params, sorted_nodes_auths, sorted_nodes_hubs)    
     

        
    
    
       
        
        
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    