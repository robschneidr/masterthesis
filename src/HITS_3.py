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
PRIVATE = 0
PUBLIC = 1



'''
notes:
    
    26.03. final step, implement semantics

    
    plotted distribution of equivalence points: it seems that the distribution is negative
    exponential. this is the kind of behavior that is wanted as a semantic value function,
    since for a search query, there will be only very few sites that make a good match, but
    exponentially more sites that do not match at all.
    
    
    
    
    compare_semantic_equivalence
    TODo: to get a more detailed model of semantic equivalence, there are two additional measures:
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
          DONE
    
    
    
    
    29.03
    
    fine-tuning of the trust change mechanics:
        there is a need to again reduce the amount of false factors, else
        the overall trust will eventually become zero and the iteration stops
        with a divide by 0 error.
        
    the question is, what behavior does one ultimately want to see?
    the increase of false factors if there is no willingness to compute! the willingness to 
    compute as the only mechanism to stop false information.
    
    --> false factor probability as an internal value of a node. it increases every time the
    node IS (?) or IS NOT (?) included in the root set and decreases every time some node discovers some other
    node's false factors.
    
    
    trust mechanics:
        instead of allowing trust in both directions, starting with 0, maybe it is more natural to start trust
        with initial value 1 and continually decrease trust when finding false factors. this also allows for a 
        proper hits calculation from the beginning
        
        
    TODO: alter the root set selection to be more precise: rank nodes according to their equivalence
    points and then decide on a threshold, from where nodes below the threshold are not taken
    into the root set.
    
    
    TODO: actualy implement get_private_ranking
    TODO: should compare_semantic_equivalence and check_public_factors be the same?
    TODO: generate a graph that suits the idea of HITS, otherwise HITS will just produce results
    as good as random when compared to the true ranking.
    TODO: hub and auth values probably have to be reset after every search query
    TODO: check for part1 and part2 if subset edge trust should really deliver different results than edge trust
        

'''


'''_______________________________UTILS_____________________________________''' 

def mean_nodes_order_similarity(nodeIDs_A, nodeIDs_B):
    
     switched_order_A = dict()
     switched_order_B = dict()
     
     #print(nodeIDs_A)
     #print(nodeIDs_B)
     
     for i in range(len(nodeIDs_A)):
         switched_order_A[nodeIDs_A[i]] = i
         switched_order_B[nodeIDs_B[i]] = i
         
     #print(switched_order_A)
     #print(switched_order_B)
     
     sum_differences = 0    
     for k, v in switched_order_A.items():
         A = v
         B = switched_order_B[k]
         sum_differences += abs(A - B)
     sum_differences /= len(switched_order_A)
     
     return sum_differences

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

def get_sorted_nodeIDs_no_params(sorted_nodes_auths, sorted_nodes_hubs):
    
    print()
    print([n._id for n in sorted_nodes_auths], "AUTH")
    print([n._id for n in sorted_nodes_hubs], "HUB")
    print()
    

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

def set_params(*args):
    return [arg for arg in args] 
        

def print_hubAuthTrust_values(nodes):
    for n in nodes:
        print(n._id, " hub = ", n.hub, "   auth =", n.auth)
    print()

def print_parents_children(nodes):
    for n in nodes:
        print("node ", n._id, "parents: ", [p._id for p in n.parents], ", children: ", [c._id for c in n.children], "edges: ", [(e, v) for e, v in n.edges.items()], "pub factors",  n.public_factors) 
        
        
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



'''_______________________________RANKING_____________________________________'''

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
    sumA = sum(factorsA.keys())
    sumB = sum(factorsB.keys())
    
    for factor, amount in factorsA.items():
        if factor in factorsB:
            penalty_for_often_occurring_factors = (1 - 1 / factor)   
            equivalence_points += min(amount, factorsB[factor]) * penalty_for_often_occurring_factors
            
    _scaler = math.log(max(sumA, sumB))
    penalty_for_large_prime_factors = (1 - (abs(sumA - sumB) / max(sumA, sumB))) ** (1 / _scaler)

    return equivalence_points * penalty_for_large_prime_factors

def get_ranking(nodes, query_factors, _type):
    
    equivalence_points = dict()
    for n in nodes:
        if _type == PRIVATE:
            equivalence_points[n._id] = compare_semantic_equivalence(query_factors, n.private_factors)
        else:
            equivalence_points[n._id] = compare_semantic_equivalence(query_factors, n.public_factors)
            
            #print(n._id, equivalence_points[n._id])
    #print(equivalence_points)
    equivalence_points = sorted(equivalence_points, key=equivalence_points.get)
    return equivalence_points



def get_query_factors(content_max, query_factors_scaling):
    #query_factors_content essentially creates the factors of the search query. because the search query
    #is usually by factors smaller than the object, query_factors_content needs to be scaled down.
    query_factors_content = random.randint(RANDINT_PRIMES_FLOOR, int(content_max ** (1 / query_factors_scaling)))
    query_factors = prime_factors_to_dict(prime_factors(query_factors_content))
    return query_factors

        
def get_root_set(nodes, size, query_factors):
    max_index = len(nodes) - 1
    id_set = set()
    if size > 1:
        while len(id_set) < size:
            id_set.add(random.randint(0, max_index))
    else: 
        for n in nodes:
            for f in query_factors:
                if f in n.public_factors:
                    #this is the at least one factor in query factors and public factors the same implementation
                    id_set.add(n._id)
                    
    return [nodes[_id] for _id in id_set]


def set_trust(nodes, query_factors, willingness_to_compute, learning_rate):
    
    #TODo: penalize low value false factors, because they are most likely to be 
    #added to the root set. DONE s
    
    for n in nodes:
        if random.random() < willingness_to_compute:
            for c in n.children:
               true_semantic_equivalence = compare_semantic_equivalence(query_factors, c.private_factors)
               public_semantic_equivalence = compare_semantic_equivalence(query_factors, c.public_factors)
               false_factor_difference = abs(public_semantic_equivalence - true_semantic_equivalence)
               trust_change = learning_rate * false_factor_difference
               n.edges[c._id] = max(0, min(1, n.edges[c._id] - trust_change))
               '''print("query: ", query_factors)
               print("checked child ", c._id, "content: ", c.content, "false factors: ", false_factor_difference, "tc: ", trust_change) 
               print("public factors: ", c.public_factors)
               print("private factors: ", c.private_factors)
               print()'''
               if false_factor_difference > 0:
                   nodes[c._id].public_factors = copy.deepcopy(nodes[c._id].private_factors)
               
               

def set_false_factors(nodes):
    
    for n in nodes:
        if random.random() < n.false_factor_probability:
            false_factors = prime_factors(random.randint(RANDINT_PRIMES_FLOOR, RANDINT_PRIMES_FLOOR + int(random.expovariate(1 / math.log(n.content)))))
            for false_factor in false_factors:
                if false_factor in n.public_factors:
                    n.public_factors[false_factor] += 1
                else:
                    n.public_factors[false_factor] = 1
                    

def false_factor_probability_function(x):
    return 1 - (1 / math.exp(3 * x))

def inverse_false_factor_probability_function(y):
    return -(1 / 3) * math.log(1 - y, math.e)                   
                    
def set_false_factor_probability(nodes, root_set):  
    root_set_IDs =  G.get_node_IDs(root_set)
    min_false_factor_probability = 0.01
    max_false_factor_probability = 0.99
    for n in nodes:
        if n._id in root_set_IDs:
            n.false_factor_probability = max(min_false_factor_probability, inverse_false_factor_probability_function(n.false_factor_probability))
        else:
            n.false_factor_probability = min(max_false_factor_probability, false_factor_probability_function(n.false_factor_probability))
         
    #TODO find a better way to set this probability. linearity is not suitable it seems
        
    
    
                    
'''_______________________________RANKING_____________________________________'''
                

'''_______________________________HITS_____________________________________''' 

def HITS_hubAuth_reset(nodes):
    for n in nodes:
        n.auth = 1.
        n.hub = 1.
    normalize_auths(nodes, len(nodes))
    normalize_hubs(nodes, len(nodes))      
                       

        
def HITS_init(nodes, content_max, uniform_init_edges, false_factor_probability_init):

    for n in nodes:
        n.auth = 1.
        n.hub = 1.
        G.set_content_and_private_factors(n, content_max)
        G.set_public_factors(n)
        if uniform_init_edges >= 0:
            G.set_all_edge_weights(nodes, uniform_init_edges)
        n.false_factor_probability = false_factor_probability_init
        
        
    normalize_auths(nodes, len(nodes))
    normalize_hubs(nodes, len(nodes))       

    
    
def HITS_iteration(nodes, n_search_queries, root_set_size=-1, n_steps=5,
                   enable_trust=False, content_max=1000, query_factors_scaling=1,
                   willingness_to_compute=0.5, learning_rate=0.001):
    
    
    
    for _ in range(n_search_queries):
        
        #print_parents_children(nodes)
        #print([n.public_factors for n in nodes])
        
        query_factors = get_query_factors(content_max, query_factors_scaling)
        root_set = get_root_set(nodes, root_set_size, query_factors)
        set_trust(nodes, query_factors, willingness_to_compute, learning_rate)
        set_false_factors(nodes)
        set_false_factor_probability(nodes, root_set)
        #print_hubAuthTrust_values(nodes)
        #print([n._id for n in root_set])
        #print([n.false_factor_probability for n in root_set])

        HITS_hubAuth_reset(nodes)
        for _ in range(n_steps):
            HITS_one_step(nodes, root_set, enable_trust)
            #print_hubAuthTrust_values(nodes)
        
        #print("query factors", query_factors)
        #for n in root_set:
            #print("node: ", n._id, "private: ", n.private_factors, "public: ", n.public_factors)
        true_ranking = get_ranking(root_set, query_factors, PRIVATE)
        public_ranking = get_ranking(root_set, query_factors, PUBLIC)
        #print("true", [t for t in true_ranking])
        #print("public", [t for t in public_ranking])
        sorted_nodes_auths, sorted_nodes_hubs = get_sorted_nodes(root_set)
        #get_sorted_nodeIDs_no_params(sorted_nodes_auths, sorted_nodes_hubs)
        #print("\n\n")
        
        if len(true_ranking) > 0:
            print("true vs auth", mean_nodes_order_similarity(true_ranking, [n._id for n in sorted_nodes_auths]))
            print("public vs auth", mean_nodes_order_similarity(public_ranking, [n._id for n in sorted_nodes_auths]))
            shuffled_nodes = true_ranking.copy()
            random.shuffle(shuffled_nodes)
            print("rnd vs auth", mean_nodes_order_similarity(shuffled_nodes, [n._id for n in sorted_nodes_auths]))
    
    print_hubAuthTrust_values(nodes)
    print([n.false_factor_probability for n in nodes])
    print_parents_children(nodes)
    print("avg trust: ", G.get_avg_trust(nodes))
    
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


'''_______________________________HITS_____________________________________'''

         


if __name__ == '__main__':
    
    n_steps = 20
    n_nodes = 100
    n_edges = 600
    n_users = n_nodes
    n_search_queries = 1000
    root_set_size = 5
    std_learning_rate = 0.1
    edge_init = 1.0
    false_factor_probability_init = 0.01
    
    n_search_queries_id = 0
    root_set_size_id = 1
    enable_trust_id = 2
    content_max_id = 3
    query_factors_scaling_id = 4
    willingness_to_compute_id = 5
    learning_rate_id = 6
    name_id = 7
    
    #params corresponding to the above definitions
    
    std_hits = [1, n_nodes, False, 10**10, 6, 0.1, 0.5, 0., "std HITS"]
    semantic_std_hits = [n_search_queries, -1, False, 10**10, 4, 0.01, std_learning_rate, "semantic std HITS"]
    semantic_trust_hits = [n_search_queries, -1, True, 10**10, 4, 0.1, std_learning_rate, "semantic trust HITS"]
    

    create_new_graph = True
    if create_new_graph:
        base_nodes = G.create_random_weighted_directed_document_nodes(n_nodes, n_edges)
        G.save_graph(base_nodes, "rnd_20n_60e_3")
    else:     
        base_nodes = G.load_graph("rnd_20n_60e_3")
        
    G.visualize(base_nodes)
    print_parents_children(base_nodes)
    
    params = set_params(std_hits,
                        semantic_std_hits,
                        semantic_trust_hits)
    
    params = [params[-1]]
    
    node_copies = [copy.deepcopy(base_nodes) for nodes in range(len(params))] 
    sorted_nodes_auths = []
    sorted_nodes_hubs = []

    
    for nodes, param in zip(node_copies, params):
        
        print("\n\n", param[name_id], "\n")
        #print_parents_children(nodes)
        uniform_init_edges = edge_init if param[root_set_size_id] < 0 else -1
        HITS_init(nodes, param[content_max_id], uniform_init_edges, false_factor_probability_init)
        HITS_iteration(nodes,
                       param[n_search_queries_id],
                       param[root_set_size_id],
                       n_steps, 
                       param[enable_trust_id],
                       param[content_max_id],
                       param[query_factors_scaling_id],
                       param[willingness_to_compute_id],
                       param[learning_rate_id])
        
        
        #print_parents_children(nodes)

        sorted_nodes_auth, sorted_nodes_hub = get_sorted_nodes(nodes)
        sorted_nodes_auths.append(sorted_nodes_auth)
        sorted_nodes_hubs.append(sorted_nodes_hub)

    
    sorted_nodeIDs_auths, sorted_nodeIDs_hubs = get_sorted_nodeIDs(params, sorted_nodes_auths, sorted_nodes_hubs)    
    
    
     

        
    
    
    
'''
def check_public_factors(node):
    
    #real_equivalence_points = compare_semantic_equivalence(prime_factors_to_dict(root_set), factorsB)
    
    #although not entirely correct, this saves computation time..
    #real_factors = prime_factors_to_dict(prime_factors(content))
    real_factors = node.private_factors
    
    #TODO find a measure for how much computing it needs
    n_false_factors = 0
    for public_factor, amount in node.public_factors.items():       
        if public_factor in real_factors:
            n_false_factors += abs(amount - real_factors[public_factor]) / public_factor
        else:
            n_false_factors += amount / public_factor
    #print(real_factors,"\n\n", node.public_factors,"\n\n", n_false_factors, "\n\n\n")
    return n_false_factors
'''
    
    
    
    
    