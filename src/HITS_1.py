# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:06:25 2023

@author: rob
"""

import Graph_1 as G
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import visualization as v



'''
notes:
    
    fixed initial trust values do not make sense,
    because trust just becomes a measure of overall connectedness.
    
    this also applies to the hub and authorities values. if uniformly
    initialized at 1.0, they solely become a measure of connectedness...
    
    the deciding factor is indeed the prior requried for the initialization
    
    ==> the prior is actually not the deciding factor. see node trust order
    
    
    
    01.03. change the trust mechanism to calculate average and no normalized
    observation: the average trust in the network remains the same(?) during
    the iterations
    setting(1, 1, -1):
    trust values  average out to a common value
    
    
    
    TODo: make statistics about how different trust mechanisms alter auth and hub orders. DONE
    
    
    06.03.
    specifications for expose:
        1. show hits converges to the same values regardless of rnd or 1.0 initialization
        2. trust mechanics:
            show that trust is able to alter the hits ranking
            2 different versions of trust:
                1. absolute value --> normalized similar to hits
                2. average value --> classic average
                    -> note that trust can not be determined by the node itself, but only
                    by other nodes in connection with the node (a king who calls himself a king)
        
    
    structure of the webgraph:
        internal vs external links
        https://www.contentpowered.com/blog/many-external-links-articles/
        avg somewhere around 15 external links, 100 internal links
        build webgraph accordingly
        problem: there are 1.8 billion websites on the internet, each of which has
        15 external and 100 internal links. this is not realistically simulatable
        on very small scale... with 20 nodes eg, the graph would almost be fully connected
        
        
        
    user inclusion:
        a user is a node in the network that only has outgoing connections
        
        
        
    09.03.
    
    instead of deepcopying the auth hub trust values, perhaps it is better to deepcopy the graph itself for each
    parameter setting
    
    considerations on trustworthiness: right now the trustworthiness between a connection from A to B is determined
    solely by the trustworthiness of B. how about "individual trustworthiness" of each connection. isnt this a more
    realistic concept since each node has an individual trust rating of the connection to another node?
    
    
    16.03
    
    TODo: deepcopy graph. DONE
    TODo: different connection concept. DONE
        the network connectivity is shifting from edge centric (edges contain all the information to build the graph)
        to node centric. nodes are central and contain information about its direct connectivity. edge information
        for some purposes (e.g. drawing) will be collected from the nodes with a corresponding function. this will
        help to not create inconsistencies btn. nodes and edges, as the data value will be gotten from nodes.
        
        
    17.03
    
    TODo: include HITS selection of root set. DONE
    
    in a node centric approach the actual graph object is not needed (since all the graph is, is a list of nodes)
    this saves a lot of graph.nodes access...
    
    
    18.03.
    
    implementation of the new trust conecpt:
        trust can only be given to the forward nodes, as a website generally not know what nodes point to it
        
    the computation of authority scores is unrealistic in a decentralized scenario. a website does not know
    what other websites point towards it. however, the information will eventually tickle through anyways,
    so it might not even be necessary
    
    ==> what ways does a website have to identify the website a user came from? -> referer header
                                                                
    
    19.03.
    
    included root set selection into hits
    observation: hub and authority values seem to be different for each run, when nodes are randomly selected
    for the root set. is this a good property of hits? (probably not..)
    

        
    TODO: solve normalization problem: due to the smaller (root) subsets, the normalization
        values will get smaller and disturb the overall hits ranking. imagine for example
        that conincidentally one node gets selected more often in the root sets than other nodes,
        then this node's hub and auth values will be larger due to the fact that normalization
        of smaller sets yield higher values. eg. normalize initial auth score of 1 under 5 nodes
        means 1 value of 1/5 for each node. with 20 nodes the value is 1/20
        
        ==> solved (probably) by summing the auth, hub, trust values over all the nodes of the
        network instead of only summing over subset every time.
        
    
    
    25.03.
    
    to better adapt, this HITS file will not include the second and third step of the 
    work and concentrates solely on the trust inclusion mechanisms
    
    
    
    
    

'''


def HITS_init(nodes, edge_trust=False, auth=-1, hub=-1, trust=-1):
    
        
    if auth >= 0:
        auths = [auth for i in range(len(nodes))]
        set_all_auths(nodes, auths)
    else:
        rnd_auths = [random.random() for i in range(len(nodes))]
        set_all_auths(nodes, rnd_auths)
        
    if hub >= 0:
        hubs = [hub for i in range(len(nodes))]
        set_all_hubs(nodes, hubs)
    else:
        rnd_hubs = [random.random() for i in range(len(nodes))]
        set_all_hubs(nodes, rnd_hubs)    
    
    
    if trust >= 0:
        set_edge_trusts(nodes, trust)     
        trusts = [trust for i in range(len(nodes))]
        set_all_trusts(nodes, trusts)
    #if trust is exactly -1, the random trust values from the graph initiation will be taken
    elif trust == -1:
        rnd_trusts = [random.random() for i in range(len(nodes))]
        set_all_trusts(nodes, rnd_trusts)
    else:
        set_edge_trusts(nodes)            
        rnd_trusts = [random.random() for i in range(len(nodes))]
        set_all_trusts(nodes, rnd_trusts)
  

def set_edge_trusts(nodes, val=-1):
    for n in nodes:
        for child, weight in n.edges.items():
            if val < 0:
                n.edges.update({child : random.random()})
            else:
                n.edges.update({child : val})
            
           

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
        
        
def set_all_trusts(nodes, trusts):
    for node, trust in zip(nodes, trusts):
        node.trust = trust
    #normalize_trusts(graph, sum(trusts))
        
def normalize_trusts(nodes, sum_trusts):
    for node in nodes:
        node.trust /= sum_trusts

        
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
                   trust_included=False, trust_normalized=False, edge_trust=False, edge_trust_auth=False, 
                   auth=-1, hub=-1, trust=-1):
    
    
    print_hubAuthTrust_values(nodes)
    #print_parents_children(nodes)
    
    
    for _ in range(n_search_queries):
        
        root_set_IDs = None
        root_set = get_root_set(nodes, root_set_size, root_set_IDs)
        print([n._id for n in root_set])
        #print_hubAuthTrust_values(nodes)
    
        for _ in range(n_steps):
            HITS_one_step(nodes, root_set, trust_included, trust_normalized, edge_trust, edge_trust_auth)
            #print_hubAuthTrust_values(nodes)
    
    print_hubAuthTrust_values(nodes)
    #print_parents_children(nodes)
    
    return nodes
        
        
        
        

def HITS_one_step(all_nodes, subset_nodes, trust_included, trust_normalized, edge_trust, edge_trust_auth):

    #nodes old is required so the algorithm does not take the already updated
    #values in the for loop.
    nodes_old = copy.deepcopy(all_nodes)
            
   
    for node in subset_nodes:
        
        parents = node.parents
        children = node.children
        
        if trust_included:
            if edge_trust:
                if edge_trust_auth:
                    node.auth = sum(nodes_old[p._id].hub * nodes_old[p._id].edges.get(node._id) for p in parents)
                else:
                    node.auth = sum(nodes_old[p._id].hub for p in parents)
            else:
                node.auth = sum(nodes_old[p._id].hub * nodes_old[p._id].trust for p in parents)
        else:         
            node.auth = sum(nodes_old[p._id].hub for p in parents)

        
        
        if trust_included:
            if edge_trust:
                node.hub = sum(nodes_old[c._id].auth * nodes_old[node._id].edges.get(c._id) for c in children)
            else:
                node.hub = sum(nodes_old[c._id].auth * nodes_old[c._id].trust for c in children)
                
        else:
            node.hub = sum(nodes_old[c._id].auth for c in children)

        
        
        if trust_normalized:
            sum_parents = sum(nodes_old[p._id].trust * nodes_old[p._id].hub for p in parents)
            sum_children = sum(nodes_old[c._id].trust * nodes_old[c._id].auth for c in children)
            node.trust = sum_parents + sum_children
        else:
            sum_parents = sum(nodes_old[p._id].trust for p in parents)
            sum_children = sum(nodes_old[c._id].trust for c in children)
            
            if len(parents) > 0 or len(children) > 0:
                node.trust = (sum_parents + sum_children) / (len(parents) + len(children))


    normalize_auths(subset_nodes, sum(n.auth for n in all_nodes))
    normalize_hubs(subset_nodes, sum(n.hub for n in all_nodes))
    
    if trust_normalized:
        normalize_trusts(subset_nodes, sum(n.trust for n in all_nodes))
        
    return nodes

        

def switch_nodes_order(nodeIDs):
    switched_orders = []
    for n in nodeIDs:
        switched_orders.append([0 for _ in range(len(n))])

    
    for n, sw in zip(nodeIDs, switched_orders):
        for i in range(len(n)):
            sw[n[i]] = i
        
    return switched_orders

def mean_nodes_order_similarity(nodeIDs_A, nodeIDs_B):
    
     switched_order_A = [0 for _ in range(len(nodeIDs_A))]
     switched_order_B = [0 for _ in range(len(nodeIDs_B))]
     
     print(nodeIDs_A)
     print(nodeIDs_B)
     
     for i in range(len(nodeIDs_A)):
         switched_order_A[nodeIDs_A[i]] = i
         switched_order_B[nodeIDs_B[i]] = i
         
     print(switched_order_A)
     print(switched_order_B)
     
     sum_differences = 0    
     for A, B in zip(switched_order_A, switched_order_B):
         sum_differences += abs(A - B)
     sum_differences /= len(switched_order_A)
     
     return sum_differences
     

    
def get_sorted_nodes(nodes):
    
    sorted_nodes_auth = copy.deepcopy(nodes)
    sorted_nodes_auth.sort(key=sort_by_auth)
    
    sorted_nodes_hub = copy.deepcopy(nodes)
    sorted_nodes_hub.sort(key=sort_by_hub) 

    sorted_nodes_trust = copy.deepcopy(nodes)
    sorted_nodes_trust.sort(key=sort_by_trust)
    
    return sorted_nodes_auth, sorted_nodes_hub, sorted_nodes_trust
    
    
def sort_by_trust(node):
    return node.trust

def sort_by_auth(node):
    return node.auth

def sort_by_hub(node):
    return node.hub

def get_trust_values(params):
    return [[node.trust for node in p] for p in params]
        

def get_sorted_nodeIDs(params, sorted_nodes_auths, sorted_nodes_hubs, sorted_nodes_trusts):
    
    sorted_nodeIDs_auths = []
    sorted_nodeIDs_hubs = []
    sorted_nodeIDs_trusts = []
    
    for i in range(3):
        if i == 0:
            value_name = "auth"
            vals = sorted_nodes_auths
            sorted_nodeIDs = sorted_nodeIDs_auths
            
        elif i == 1:
            value_name = "hub"
            vals = sorted_nodes_hubs
            sorted_nodeIDs = sorted_nodeIDs_hubs
        else:
            value_name = "trust"
            vals = sorted_nodes_trusts
            sorted_nodeIDs = sorted_nodeIDs_trusts
                      
        for i, param in enumerate(params):
            sorted_nodeIDs.append([node._id for node in vals[i]])
            print([node._id for node in vals[i]], "node " + value_name + " order " + param[name_id])
            
        print()
        
    print()
            
    return sorted_nodeIDs_auths, sorted_nodeIDs_hubs, sorted_nodeIDs_trusts



    
    



'''_______________________________UTILS_____________________________________'''  

def set_params(*args):
    return [arg for arg in args] 
        

def print_hubAuthTrust_values(nodes):
    for n in nodes:
        print("node", n._id, " hub = ", n.hub, "   auth =", n.auth, " node_trust = ", n.trust)
    print()

def print_parents_children(nodes):
    for n in nodes:
        print("node ", n._id, "parents: ", [p._id for p in n.parents], ", children: ", [c._id for c in n.children], "edges: ", [(e, v) for e, v in n.edges.items()])   
        
    print()


def plot_node_rankings(params, sorted_nodeIDs_auths, sorted_nodeIDs_hubs, sorted_nodeIDs_trusts):
    x = np.arange(0, len(sorted_nodeIDs_auths[0]))   
    fig, axs = plt.subplots(3, 1)
    for ID, param in zip(sorted_nodeIDs_auths, params): 
        #print(ID, param[name_id])
        axs[0].plot(x, ID, label=param[name_id])

    axs[0].set_ylabel('Authority')
    #axs[0].grid(True)
    
    for ID, param in zip(sorted_nodeIDs_hubs, params): 
        #print(ID, param[name_id])
        axs[1].plot(x, ID, label=param[name_id])
    axs[1].set_ylabel('Hub')
    #axs[1].grid(True)
    
    for ID, param in zip(sorted_nodeIDs_trusts, params): 
        #print(ID, param[name_id])
        axs[2].plot(x, ID, label=param[name_id])
    axs[2].set_xlabel('Node Ranking from Lowest to Highest')
    axs[2].set_ylabel('Trust')
    #axs[2].grid(True)
    
    fig.tight_layout()
    plt.legend(loc='best', bbox_to_anchor=(1.2, 1))
    plt.show()
    
    
'''_______________________________UTILS_____________________________________'''
    


if __name__ == '__main__':
    
    n_steps = 20
    n_nodes = 20
    n_edges = 80
    n_users = n_nodes
    n_known_nodes = 5
    n_search_queries = 500
    
    n_search_queries_id = 0
    root_set_size_id = 1
    trust_included_id = 2
    trust_normalized_id = 3
    edge_trust_id = 4
    edge_trust_auth_id = 5
    auth_id = 6
    hub_id = 7
    trust_id = 8
    name_id = 9
    
    
    
    #params corresponding to the above definitions
    std_HITS_param = [1, n_nodes, False, False, False, False, 1, 1, 1, "Standard"]
    std_HITS_rndHubAuth_param = [1, n_nodes, False, False, False, False, -1, -1, 1, "Standard, Rnd Hub/Auth"]
    trust_HITS_normalized = [1, n_nodes, True, True, False, False, 1, 1, 1, "Normalized Trust"]
    rnd_trust_HITS_normalized = [1, n_nodes, True, True, False, False, 1, 1, -1, "Normalized, Rnd Trust"]
    rndTrust_rndHits_normalized = [1, n_nodes, True, True, False, False, -1, -1, -1, "Normalized Rnd Trust/Hub/Auth"]
    trust_HITS_avg = [1, n_nodes, True, False, False, False, 1, 1, 1, "Averaged Trust"]
    rnd_trust_HITS_avg = [1, n_nodes, True, False, False, False, 1, 1, -1, "Averaged Rnd Trust"]
    edge_trust_HITS = [1, n_nodes, True, False, True, False, 1, 1, -1, "Edge Trust"]
    edge_trust_auth_HITS = [1, n_nodes, True, False, True, True, 1, 1, -1, "Edge Trust with Authority"]
    subset_edge_trust_auth = [n_search_queries, int(n_nodes/3), True, False, True, True, 1, 1, -1, "Subset Edge Trust"]

    
    
    #base_nodes = G.create_random_weighted_directed_document_nodes(n_nodes, n_edges)
    base_nodes = G.load_graph("rnd_20n_60e_1")
    G.visualize(base_nodes)
    print_parents_children(base_nodes)
    #G.save_graph(base_nodes, "rnd_20n_60e_1")
    
    params = set_params(std_HITS_param,
                        std_HITS_rndHubAuth_param, 
                        trust_HITS_normalized, 
                        rnd_trust_HITS_normalized,  
                        rndTrust_rndHits_normalized,  
                        trust_HITS_avg,
                        rnd_trust_HITS_avg,
                        edge_trust_HITS,
                        edge_trust_auth_HITS,
                        subset_edge_trust_auth)
    
    #params = [params[0], params[-1]]
    #params = [params[2]]
    
    node_copies = [copy.deepcopy(base_nodes) for nodes in range(len(params))]
    
    
    sorted_nodes_auths = []
    sorted_nodes_hubs = []
    sorted_nodes_trusts = []
    unsorted_nodes = []
    
    
    for nodes, param in zip(node_copies, params):
        
        print("\n\n", param[name_id], "\n")
        #print_parents_children(nodes)
        HITS_init(nodes, param[edge_trust_id], param[auth_id], param[hub_id], param[trust_id])
        
        HITS_iteration(nodes,
                       param[n_search_queries_id],
                       param[root_set_size_id],
                       n_steps, 
                       param[trust_included_id],
                       param[trust_normalized_id],
                       param[edge_trust_id],
                       param[edge_trust_auth_id],
                       param[auth_id],
                       param[hub_id],
                       param[trust_id])
        

        
        unsorted_nodes.append(nodes)
        sorted_nodes_auth, sorted_nodes_hub, sorted_nodes_trust = get_sorted_nodes(nodes)
        sorted_nodes_auths.append(sorted_nodes_auth)
        sorted_nodes_hubs.append(sorted_nodes_hub)
        sorted_nodes_trusts.append(sorted_nodes_trust)
        
      
    
    
    
    
    
    sorted_nodeIDs_auths, sorted_nodeIDs_hubs, sorted_nodeIDs_trusts = get_sorted_nodeIDs(params, sorted_nodes_auths, sorted_nodes_hubs, sorted_nodes_trusts)    
    
    #v.heatmap_auth_rankings(sorted_nodeIDs_auths, [param[name_id] for param in params])
    v.heatmap_hub_rankings(sorted_nodeIDs_hubs, [param[name_id] for param in params])
    #v.heatmap_adjacency_matrix(nodes)
    #v.heatmap_trusts(get_trust_values(unsorted_nodes[5:7]), [param[name_id] for param in params[5:7]])
    #v.heatmap_trusts(get_trust_values(unsorted_nodes), [param[name_id] for param in params])
    
    
    #plot_node_rankings(params, sorted_nodeIDs_auths, sorted_nodeIDs_hubs, sorted_nodeIDs_trusts)        
    #diff = mean_nodes_order_similarity([node._id for node in sorted_nodes_auths[0]], [node._id for node in sorted_nodes_auths[1]])
    #print(diff)    
        
    
    
       
        
        
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    