# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:50:40 2023

@author: rob
"""

import Graph_2 as G
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import visualization as v

np.set_printoptions(precision=3, suppress=True)



'''
notes:
    
    

        
    TODO: solve normalization problem: due to the smaller (root) subsets, the normalization
        values will get smaller and disturb the overall hits ranking. imagine for example
        that conincidentally one node gets selected more often in the root sets than other nodes,
        then this node's hub and auth values will be larger due to the fact that normalization
        of smaller sets yield higher values. eg. normalize initial auth score of 1 under 5 nodes
        means 1 value of 1/5 for each node. with 20 nodes the value is 1/20
        
        ==> solved (probably) by summing the auth, hub, trust values over all the nodes of the
        network instead of only summing over subset every time.
        the small ranking discrepancies btn std hits and std hits user engagement very likely
        come from the fact that a, h, t values of the user nodes are also taken into account.
        
        normalization issues.. as the subsets are not as large as the full set, normalized values
        of subsets will always be smaller than full set. this becomes an issue especially when subsets
        have different sizes...
        -> there may be an incompatibality between the 0 to 1 trust values and the normalized values. perhaps
        it will be necessary to average the hub and auth values instead of normalizing them...
        
    
    
    
    25.03.
    
    TODo: in order to simplify inclusion of the 2nd step, the 1st step needs to be significantly simplified
    
    as a way to achieve this, many of the hyperparameters of the 1st stept are cut out. 
    DONE
    
    
    
    TODo: user inclusion update
    
    instead of including the users in the document graph, make them external
    DONE
    
    
    
    TOPIC: do the local HITS approximations find the global optimum for hits?
    
    
    
    
    
    

'''


def HITS_init(nodes, auth=-1, hub=-1):

    for n in nodes:
        n.auth = 1.
        n.hub = 1.
    normalize_auths(nodes, len(nodes))
    normalize_hubs(nodes, len(nodes))
    
           

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
                   enable_trust=False, users=None, 
                   auth=-1, hub=-1, trust=-1):
    
    avg_trusts = []
    user_trusts = []
    #avg_trust = G.get_avg_trust(nodes)
    
    #print("avg trust", avg_trust)
    #print_hubAuthTrust_values(nodes)
    #print_parents_children(nodes)
    
    
    for _ in range(n_search_queries):
        
        root_set_IDs = None
        root_set = get_root_set(nodes, root_set_size, root_set_IDs)
        #print([n._id for n in root_set])
        #print_hubAuthTrust_values(nodes)
        
        if users is not None:
            selected_user = G.select_rnd_user(users)
            selected_user.adjust_children(root_set)
            avg_trusts.append(G.get_avg_trust(nodes))
            user_trusts.append(G.get_users_avg_trust(users))

            print("user ", selected_user._id, "children: ", [c._id for c in selected_user.children])
            print("user avg trust", G.get_users_avg_trust(users))
            print()
    
        for _ in range(n_steps):
            HITS_one_step(nodes, root_set, enable_trust, users)
            #print_hubAuthTrust_values(nodes)
    
    print_hubAuthTrust_values(nodes)
    #print_parents_children(nodes)
    
    return nodes, avg_trusts, user_trusts
        
        
        
        

def HITS_one_step(all_nodes, subset_nodes, enable_trust, users):
    
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



def get_user_connection_count(nodes, users):
    counts = [0 for _ in range(len(nodes))]
    for u in users:
        for c in u.children:
            counts[c._id] += 1
    return counts
    



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
    
    
    
'''_______________________________UTILS_____________________________________'''
    


if __name__ == '__main__':
    
    n_steps = 20
    n_nodes = 20
    n_edges = 80
    n_users = n_nodes
    n_search_queries = 100
    root_set_size = 6
    
    n_search_queries_id = 0
    root_set_size_id = 1
    enable_trust_id = 2
    users_engaged_id = 3
    name_id = 4
    
    
    
    #params corresponding to the above definitions
    
    std_hits = [1, n_nodes, False, False, "std HITS, no users"]
    std_subset_hits = [n_search_queries, root_set_size, False, False, "std subset HITS, no users"]
    std_hits_trust = [n_search_queries, root_set_size, True, False, "std HITS with trust model, no users"]
    user_hits_trust = [n_search_queries, root_set_size, True, True, "user HITS with trust model"]
    user_hits = [n_search_queries, root_set_size, False, True, "user HITS"]
    
    
    
    #base_nodes = G.create_random_weighted_directed_document_nodes(n_nodes, n_edges)
    base_nodes = G.load_graph("rnd_20n_60e_1")
    #G.visualize(base_nodes)
    print_parents_children(base_nodes)
    #G.save_graph(base_nodes, "rnd_20n_60e_2")
    
    params = set_params(std_hits,
                        std_subset_hits,
                        std_hits_trust,
                        user_hits_trust,
                        user_hits)
    
    #params = [params[3]]
    
    
    params = [params[-2]]
    
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
                       param[enable_trust_id],
                       None)
        
        if param[users_engaged_id]:
            users = G.create_users(n_users)
            _, avg_trusts, user_trusts = HITS_iteration(nodes,
                                         param[n_search_queries_id],
                                         param[root_set_size_id],
                                         n_steps, 
                                         param[enable_trust_id],
                                         users)
        
        
        #print_parents_children(nodes)

        sorted_nodes_auth, sorted_nodes_hub = get_sorted_nodes(nodes)
        sorted_nodes_auths.append(sorted_nodes_auth)
        sorted_nodes_hubs.append(sorted_nodes_hub)
        #v.plot_avg_vs_user_trusts(user_trusts)
        #v.draw_network_with_users(nodes, users)
        #print(get_user_connection_count(nodes, users))
        v.heatmap_node_user_adjacency_matrix(nodes, users)
        #v.heatmap_adjacency_matrix(nodes)
    
    sorted_nodeIDs_auths, sorted_nodeIDs_hubs = get_sorted_nodeIDs(params, sorted_nodes_auths, sorted_nodes_hubs)    
     

        
    
    
       
        
        
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    