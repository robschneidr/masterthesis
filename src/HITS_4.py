# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 09:59:26 2023

@author: rob
"""


import Graph_4 as G
import Functions as F
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import math
import visualization as vis
from collections import deque


RANDINT_PRIMES_FLOOR = 2
FALSE_FACTOR_FLOOR = 0.01
FALSE_FACTOR_CEIL = 0.99
PRIVATE = 0
PUBLIC = 1



'''
notes:
    
    
    TODO: generate a graph that suits the idea of HITS, otherwise HITS will just produce results
    as good as random when compared to the true ranking.
    TODO: eliminate arbitrary learning rate. forgot the concept....
    
    TODO: Hyperparameter Elimination Paradigm
    
    TODO: make the graph dynamic again. make it possible for nodes to change their connection based on hub/auth values
          
    17.04. performing experiments with the graph creation process:
        what is the more probable way that websites connect? do they connect to other websites that they share
        semantic information with, or do they connect to websites that are entirely different than themselves?
        check the 2 graphs "Difference_between_HITS_and_Semantic_Rankings_after_Semantic_Graph_Building" and
        "Difference_between_HITS_and_Semantic_Rankings_after_Semantic_Graph_Building_with_Ranking_Reversed".
        they do not differ a lot from each other and they also do not differ too much from the 
        "Difference_between_HITS_and_Semantic_Rankings" Experiment from HITS_3.
        
        the implications of this are not yet entirely clear. 
        
        another issue found: because of the semantic graph connection, some nodes do not have parents, 
        resulting in possibly many auth scores of 0.
        
        
        --> the reason the difference between auth and private ranking is not becoming zero is the fact that
        both auth scores can get 0 and equivalence points also can be 0. so if there are for example 10 nodes
        in the root set that both yield an auth score and equivalence points score of 0, the ranking will be
        arbitrary.
        
        
        solutions:
            1) equivalence points in compare_semantic_equivalence is set to 1 instead 0 to still profit from
                the multiplicative penalty factors
            2) implement dynamic graph mechanics, where nodes without parents get deleted and connections below
                a certain trust level also get deleted
                
        ==> this can be interpreted as websites that actually (private) contain computationally irreducible gibberish
        will eventually be removed, since their only way of getting into search results is by getting false factors
        ==> this also hints at the fact that humanly comprehensible information must be compressible. it can and must
        consist of smaller prime objects, but ultimately human interpretable = compressible = composed
        
    
    
    18.04
    
    work on the concept of setting the trust change after discovering a false factor probability. discovering the first lie
    is probably the biggest change in information, so it should be valued the most. 
    
    concept: functions also need some kind of hyperparameter (steepness) that is arbitrarily selected. why not decentrally
    select hyperparamters that are a result of the underlying structures and their interests.
    => solution: get a value that has combined information over the whole network and use this value as a reference to
    determine function hyperparameters. see implementation
    
    
    TODO: refine the set false factor probability function. the problem right now is that at some point all the nodes
    are in the root set, and no node has the need to set false factors. however, the ranking is quite large then and
    this is not good. better to set false factor probability based on the actual ranking, or 0 if it is not even included.
    
        

'''


'''_______________________________UTILS_____________________________________''' 

def mean_nodes_order_similarity(nodeIDs_A, nodeIDs_B):
    
     switched_order_A = dict()
     switched_order_B = dict()
     
     
     for i in range(len(nodeIDs_A)):
         switched_order_A[nodeIDs_A[i]] = i
         switched_order_B[nodeIDs_B[i]] = i
     
     sum_differences = 0    
     for k, v in switched_order_A.items():
         A = v
         B = switched_order_B[k]
         sum_differences += abs(A - B)
     sum_differences /= len(switched_order_A)
     
     return sum_differences / len(nodeIDs_A)

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
    
    sorted_nodes_auth = copy.copy(nodes)
    sorted_nodes_auth.sort(key=sort_by_auth)
    
    sorted_nodes_hub = copy.copy(nodes)
    sorted_nodes_hub.sort(key=sort_by_hub) 
    
    return sorted_nodes_auth, sorted_nodes_hub


def sort_by_auth(node):
    return node.auth

def sort_by_hub(node):
    return node.hub

def print_hubAuth_Ranking(sorted_nodes_auths, sorted_nodes_hubs):
    
    print()
    print([n._id for n in sorted_nodes_auths], "AUTH")
    print([n._id for n in sorted_nodes_hubs], "HUB")
    print()
    

def set_params(*args):
    return [arg for arg in args] 
        

def print_hubAuth_values(nodes):
    for n in nodes:
        print(n._id, " hub = ", n.hub, "   auth =", n.auth)
    print()

def print_parents_children(nodes):
    for n in nodes:
        print("node ", n._id, "parents: ", [p._id for p in n.parents], ", children: ", [c._id for c in n.children], "edges: ", [(e, v) for e, v in n.edges.items()], "pub factors",  n.public_factors) 
        
        
    print()
 
    

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
    
    

    sumA = sum(factorsA.keys())
    sumB = sum(factorsB.keys())
    #make it 1, so it can still take the multiplicative penalties into account
    equivalence_points = 1
    
    for factor, amount in factorsA.items():
        if factor in factorsB:
            penalty_for_often_occurring_factors = (1 - 1 / factor)   
            equivalence_points += min(amount, factorsB[factor]) * penalty_for_often_occurring_factors
            
    _scaler = math.log(max(sumA, sumB))
    penalty_for_large_prime_factors = (1 - (abs(sumA - sumB) / max(sumA, sumB))) ** (1 / _scaler)
    
    #print("eq pen", equivalence_points * penalty_for_large_prime_factors)
    return equivalence_points * penalty_for_large_prime_factors



'''
see notes 17.04
'''
def get_ranking(ids_and_factors, query_factors):
    
    equivalence_points = dict()
    for _id, factors in ids_and_factors:
        equivalence_points[_id] = compare_semantic_equivalence(query_factors, factors)

            
        #print(_id, equivalence_points[_id])
    #print(equivalence_points)
    #this makes sure that nodes that share 0 equivalence are shuffled for the ranking,
    #as doing otherwise would skew the rankings.
    n_zeros = 0
    for e in equivalence_points.values():
        if e == 0:
            n_zeros +=1
    equivalence_points = sorted(equivalence_points, key=equivalence_points.get)
    shuffled_part = equivalence_points[:n_zeros]
    random.shuffle(shuffled_part)
    equivalence_points[:n_zeros] = shuffled_part
    
    return equivalence_points




def get_query_factors(content_max, query_factors_scaling):
    #query_factors_content essentially creates the factors of the search query. because the search query
    #is usually by factors smaller than the object, query_factors_content needs to be scaled down.
    query_factors_content = random.randint(RANDINT_PRIMES_FLOOR, int(content_max ** (1 / query_factors_scaling)))
    query_factors = prime_factors_to_dict(prime_factors(query_factors_content))
    return query_factors

        
def get_root_set(nodes, query_factors):
    id_set = set()
    for n in nodes:
        for f in query_factors:
            if f in n.public_factors:
                #this is the at least one factor in query factors and public factors the same implementation
                id_set.add(n._id)
                    
    return [nodes[_id] for _id in id_set]


def compute(nodes, query_factors, willingness_to_compute, false_factor_levels, edge_init):
    #this method should resemble the "additional" energy that a node decides to spend
    #on contributing to the network. for now (18.04) the additional computational work
    #is done through the calculation of the private_semantic_equivalence relative to the query.
    computed_trusts = []
    for n in nodes:
        if random.random() < willingness_to_compute:
            computed_trusts.append(set_trust(n , query_factors, willingness_to_compute, false_factor_levels, edge_init))
    
    if computed_trusts:
        #print("compo", computed_trusts, np.mean(computed_trusts))
        return np.mean(computed_trusts)
    else:
        return edge_init
            
            


def set_trust(node, query_factors, willingness_to_compute, false_factor_levels, edge_init):
    
    new_trust = edge_init  
    for c in node.children:
        #Note that the access to c's private factors can only be obtained through computation of the factors
       private_semantic_equivalence = compare_semantic_equivalence(query_factors, c.private_factors)
       public_semantic_equivalence = compare_semantic_equivalence(query_factors, c.public_factors)
       false_factor_level = abs(public_semantic_equivalence - private_semantic_equivalence)
       mean_false_factor_level = np.mean(false_factor_levels)

       previous_trust = F.ilffd(node.edges[c._id])
       trust_change = mean_false_factor_level - false_factor_level
       new_trust = F.lffd(previous_trust + trust_change)

       node.edges[c._id] = new_trust

       if false_factor_level > 0:
           G.reset_public_factors(nodes[c._id])
           
       false_factor_levels.append(false_factor_level)
       
    return new_trust
               
               

def set_false_factors(nodes): 
    for n in nodes:
        if random.random() < n.false_factor_probability:
            false_factors = prime_factors(random.randint(RANDINT_PRIMES_FLOOR, RANDINT_PRIMES_FLOOR + int(random.expovariate(1 / math.log(n.content)))))
            for false_factor in false_factors:
                if false_factor in n.public_factors:
                    n.public_factors[false_factor] += 1
                else:
                    n.public_factors[false_factor] = 1
                    

    
                    
def set_false_factor_probability(nodes, root_set):  
    root_set_IDs =  G.get_node_IDs(root_set)
    for n in nodes:
        if n._id in root_set_IDs:
            n.false_factor_probability = max(FALSE_FACTOR_FLOOR, F.inverse_false_factor_probability_function(n.false_factor_probability))
        else:
            n.false_factor_probability = min(FALSE_FACTOR_CEIL, F.false_factor_probability_function(n.false_factor_probability))
         
    
    
def get_n_false_factors(nodes):
    return sum(abs(len(n.public_factors) - len(n.private_factors)) for n in nodes)
        


        
        
        
    
    
                    
'''_______________________________RANKING_____________________________________'''
                

'''_______________________________HITS_____________________________________''' 

def HITS_hubAuth_reset(nodes):
    for n in nodes:
        n.auth = 1.
        n.hub = 1.
    normalize_auths(nodes, len(nodes))
    normalize_hubs(nodes, len(nodes))      
                       
       
def HITS_init(nodes, content_max, edge_init, false_factor_levels):
    
    HITS_hubAuth_reset(nodes)
    for n in nodes:
        G.init_content(n, content_max)
        if edge_init >= 0:
            G.set_all_edge_weights(nodes, edge_init)
    
    #this is important for the np.mean in compute_semantic_equivalence to be not NaN at first
    false_factor_levels.append(0)
 
    
def HITS_iteration(nodes, n_search_queries, n_steps=5,
                   content_max=1000, query_factors_scaling=1,
                   willingness_to_compute=0.5, edge_init=0.5):
    
    false_factor_levels = deque(maxlen=len(nodes))
    HITS_init(nodes, content_max, edge_init, false_factor_levels)
      
    order_similarities_rnd_auth = []
    order_similarities_private_auth = []
    order_similarities_public_auth = []
    avg_false_factor_probabilities = []
    avg_trusts = []
    
    
    
    for nth_query in range(n_search_queries):
        
        #print("nth query: ", nth_query)
        #print_parents_children(nodes)
        
        query_factors = get_query_factors(content_max, query_factors_scaling)
        avg_computed_trust = compute(nodes, query_factors, willingness_to_compute, false_factor_levels, edge_init)

        
        n_lost_edges = G.replace_parentless_nodes(nodes, content_max)
        n_removed_edges = G.remove_untrustworthy_edges(nodes, avg_computed_trust, edge_init)
        G.replace_edges(nodes, n_lost_edges + n_removed_edges, edge_init)

        root_set = get_root_set(nodes, query_factors)
        #print("n edges in the system AFTER: ", G.get_n_edges(nodes))
        
        
        set_false_factors(nodes)
        set_false_factor_probability(nodes, root_set)
        #print("avg false factor probs:", G.get_avg_false_factor_probability(nodes))
        

  
        HITS_hubAuth_reset(nodes)
        for _ in range(n_steps):
            HITS_one_step(nodes, root_set)
            
        
            
        private_ranking = get_ranking([(n._id, n.private_factors) for n in root_set], query_factors)
        public_ranking = get_ranking([(n._id, n.public_factors) for n in root_set], query_factors)
        sorted_nodes_auths, sorted_nodes_hubs = get_sorted_nodes(root_set)
        sorted_nodes_auths_IDs = [n._id for n in sorted_nodes_auths]
        sorted_nodes_hubs_IDs = [n._id for n in sorted_nodes_hubs]
        avg_trusts.append(G.get_avg_trust(nodes))
        avg_false_factor_probabilities.append(G.get_avg_false_factor_probability(nodes))
        
        if len(private_ranking) > 0:
            shuffled_nodes = copy.copy(private_ranking)
            random.shuffle(shuffled_nodes)
            order_similarity_rnd_auth = mean_nodes_order_similarity(shuffled_nodes, sorted_nodes_auths_IDs)
            order_similarity_private_auth = mean_nodes_order_similarity(private_ranking, sorted_nodes_auths_IDs)
            order_similarity_public_auth = mean_nodes_order_similarity(public_ranking, sorted_nodes_auths_IDs)
            order_similarities_rnd_auth.append(order_similarity_rnd_auth)
            order_similarities_private_auth.append(order_similarity_private_auth)
            order_similarities_public_auth.append(order_similarity_public_auth)
            
            if nth_query % 50 == 0:
                print("query factors: ", query_factors)
                print("private vs auth", order_similarity_private_auth)
                print("public vs auth", order_similarity_public_auth)
                print("rnd vs auth", order_similarity_rnd_auth)  
                print("private: ", private_ranking)
                print("public: ", public_ranking)
                print("hits auth: ", sorted_nodes_auths_IDs)
                print("hits hubs: ", sorted_nodes_hubs_IDs)
        
        if nth_query % 50 == 0:
            print_hubAuth_values(root_set)
            print_parents_children(nodes)
            print_hubAuth_Ranking(sorted_nodes_auths, sorted_nodes_hubs)
            print(np.mean(false_factor_levels))
            print("avg trust", np.mean(avg_trusts))
            print("n parentless: ", G.get_n_parentless(nodes), "lost edges:", n_lost_edges, "n removed edges: ", n_removed_edges, "avg trust:" )

          
        '''print("query factors: ", query_factors)
        print("private: ", [(n._id, n.private_factors) for n in root_set])
        print("pulic: ", [(n._id, n.public_factors) for n in root_set])
        private_ranking = get_ranking([(n._id, n.private_factors) for n in root_set], query_factors)
        public_ranking = get_ranking([(n._id, n.public_factors) for n in root_set], query_factors)
        print("private ranking: ", private_ranking)
        print("public ranking: ", public_ranking)
        print()'''

    
    #vis.plot_order_similarities(order_similarities_rnd_auth, order_similarities_private_auth, order_similarities_public_auth)
    #vis.plot_avg_trusts(avg_trusts)
    #vis.plot_avg_false_factor_probabilities(avg_false_factor_probabilities)
    return nodes
        
        
        
        

def HITS_one_step(all_nodes, subset_nodes):
    
    #nodes old is required so the algorithm does not take the already updated
    #values in the for loop.
    nodes_old = copy.deepcopy(all_nodes)
            
    for node in subset_nodes:
        node.auth = sum(nodes_old[p._id].hub * nodes_old[p._id].edges.get(node._id) for p in node.parents)
        node.hub = sum(nodes_old[c._id].auth * nodes_old[node._id].edges.get(c._id) for c in node.children)

    normalize_auths(subset_nodes, sum(n.auth for n in all_nodes))
    normalize_hubs(subset_nodes, sum(n.hub for n in all_nodes))
    
    return nodes


'''_______________________________HITS_____________________________________'''

         


if __name__ == '__main__':
    
    n_nodes = 100
    n_edges = 400

    n_search_queries = 3000
    n_steps = 20
    content_max = 10**6
    query_factors_scaling = 3
    willingness_to_compute = 0.2
    edge_init = 0.5



    create_new_graph = True
    if create_new_graph:
        #base_nodes = G.create_random_weighted_directed_document_nodes(n_nodes, n_edges)
        nodes = G.create_semantic_connection_nodes(n_nodes, n_edges, content_max)
        G.save_graph(nodes, "rnd_100n_400e_4")
    else:     
        nodes = G.load_graph("rnd_100n_400e_4")
        
    #G.visualize(base_nodes)
    
    print_parents_children(nodes)

    HITS_iteration(nodes,
                   n_search_queries,
                   n_steps, 
                   content_max,
                   query_factors_scaling,
                   willingness_to_compute,
                   edge_init)
        
        
    #print_parents_children(nodes)

    
     










    
    
    
    
    