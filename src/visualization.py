# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:30:10 2023

@author: rob
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import Graph_2 as G
import HITS_3 as H3
import HITS_5 as H
import networkx as nx
import random
import copy
import math
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import skewnorm

def plot_estimated_kolmogorov(nodes):
    
    
    # Generate example datasets
    data1 = [H.get_kolmogorov_compression_ratio(H.dict_factors_to_list(n.private_factors))[1] for n in nodes if n.false_factor_probability < 0.01]
    data2 = [H.get_kolmogorov_compression_ratio(H.dict_factors_to_list(n.private_factors))[1] for n in nodes if (n.false_factor_probability >= 0.01 and n.false_factor_probability <= 0.98 and H.get_kolmogorov_compression_ratio(H.dict_factors_to_list(n.private_factors))[1] < 300)]
    data3 = [H.get_kolmogorov_compression_ratio(H.dict_factors_to_list(n.private_factors))[1] for n in nodes if (n.false_factor_probability > 0.98 and H.get_kolmogorov_compression_ratio(H.dict_factors_to_list(n.private_factors))[1] < 300)]
    
    
    
    print("d", data1, data2, data3)
    # Fit a normal distribution to each dataset
    params1 = norm.fit(data1)
    params2 = norm.fit(data2)
    params3 = norm.fit(data3)
    
    print("p", params1, params2, params3)
    
    # Plot histograms for each dataset
    #plt.hist(data1, density=True, alpha=0.5, label='Dataset 1')
    #plt.hist(data2, density=True, alpha=0.5, label='Dataset 2')
    #plt.hist(data3, density=True, alpha=0.5, label='Dataset 3')
    
    # Plot estimated normal distribution curves
    x = np.linspace(0, 200, 1000)
    
    pdf1 = skewnorm.pdf(x, params1[0], params1[1])
    pdf2 = skewnorm.pdf(x, params2[0], params2[1])
    pdf3 = skewnorm.pdf(x, params3[0], params3[1])
    
    
    
    
    plt.plot(x, pdf1, color='red', linewidth=2, label='FFP < 0.01')
    plt.plot(x, pdf2, color='blue', linewidth=2, label='0.01 <= FFP <= 0.98')
    plt.plot(x, pdf3, color='green', linewidth=2, label='FFP > 0.98')
    
    # Add legend and labels
    plt.legend()
    plt.xlabel('Estimation of Kolmogorov Complexity')
    plt.ylabel('Probability Density')
    
    # Display the plot
    plt.show()


def plot_FFP_distribution2(ffps):
    plt.rcParams['figure.dpi'] = 600
    
    # Create a list of 100 random numbers between 0 and 1
    data = ffps
    mean = np.mean(ffps)
    var = np.var(data)
    
    ax = sns.histplot(data, bins=20, kde=True)
    
    # Set the plot title and axis labels
    plt.title("FFP Distribution")

    plt.xlabel('FFP')
    plt.ylabel("Number of Nodes")
    custom_label_1 = "Mean = " + str(round(mean, 2))
    custom_label_2 = "Variance = " + str(round(var, 2))
    
    ax.text(0.3, 0.9, custom_label_1, transform=ax.transAxes, fontsize=12, color='black')
    ax.text(0.3, 0.85, custom_label_2, transform=ax.transAxes, fontsize=12, color='black')

    
    # Show the plot
    plt.show()

def plot_FFP_distribution(ffps):
    plt.rcParams['figure.dpi'] = 600
    data = ffps
    mean = np.mean(ffps)
    var = np.var(data)
    
    print("data: ", data)
    # Categorize the data into three categories
    less_than_001 = [d for d in data if d < 0.01]
    between_001_and_099 = [d for d in data if (d >= 0.01 and d <=0.99)]
    greater_than_099 = [d for d in data if d > 0.99]
    
    # Create a list of the categories and their counts
    categories = ["FFP < 0.01", "0.01 <= FFP <= 0.99", "FFP > 0.99"]
    values = [len(less_than_001), len(between_001_and_099), len(greater_than_099)]
    data_dict = {"Category": categories, "Count": values}
    
    # Create a bar plot using Seaborn
    sns.barplot(x="Category", y="Count", data=data_dict)
    
    plt.ylabel("Number of Nodes")
    # Set the plot title
    plt.title("FFP Distribution, Mean = " + str(round(mean, 2)) + ", Variance = " + str(round(var, 2)))
    
    # Show the plot
    plt.show()




def plot_trust_wtc_ffp(trusts, wtcs, ffps):
    avg_trusts = []
    avg_wtcs = []
    avg_ffps = []
    size = len(trusts)
    

    
    step_size = 20
    for i in range(0, size - step_size - 1, step_size):
        avg_trusts.append(np.mean(trusts[i:i + step_size]))
        avg_wtcs.append(np.mean(wtcs[i:i + step_size]))
        avg_ffps.append(np.mean(ffps[i:i + step_size]))
    plt.rcParams['figure.dpi'] = 600
    # create the line plot
    #x = np.arange(len(all_parentless))
    x = np.arange(len(avg_trusts))

    sns.regplot(x=x, y=avg_trusts, label="Trust", scatter=True)
    sns.regplot(x=x, y=avg_wtcs, label="WTC", scatter=True)
    sns.regplot(x=x, y=avg_ffps, label="FFP", scatter=True)
    
    # set the plot title and axes labels
    plt.title("Dynamics of Trust, WTC and False Factor Probability")
    plt.xlabel("Iterations")
    plt.ylabel("Average Quantity")
    plt.legend()
    # show the plot
    plt.show()

def plot_parentless_and_removed_edges(all_parentless, all_removed_edges):
    
    avg_parent = []
    avg_edge = []
    size = len(all_parentless)

    
    step_size = 50
    for i in range(0, size - step_size - 1, step_size):
        avg_parent.append(np.mean(all_parentless[i:i + step_size]))
        avg_edge.append(np.mean(all_removed_edges[i:i + step_size]))
    plt.rcParams['figure.dpi'] = 600
    # create the line plot
    #x = np.arange(len(all_parentless))
    x = np.arange(len(avg_parent))

    sns.regplot(x=x, y=avg_parent, label="Parentless Nodes", scatter=True)
    sns.regplot(x=x, y=avg_edge, label="Untrustworthy Edges", scatter=True)
    
    # set the plot title and axes labels
    plt.title("Dynamics of the Document Nodes")
    plt.xlabel("Iterations")
    plt.ylabel("Quantity")
    plt.legend()
    
    # show the plot
    plt.show()
    


def plot_parentless_and_removed_and_wtc(all_parentless, all_removed_edges, wtcs):
    avg_parent = []
    avg_edge = []
    avg_wtcs = []
    size = len(all_parentless)

    
    step_size = 50
    for i in range(0, size - step_size - 1, step_size):
        avg_parent.append(np.mean(all_parentless[i:i + step_size]))
        avg_edge.append(np.mean(all_removed_edges[i:i + step_size]))
        avg_wtcs.append(np.mean(wtcs[i:i + step_size]))
    

    
    
    plt.rcParams['figure.dpi'] = 600
    sns.regplot(x=avg_wtcs, y=avg_parent, label="Parentless Nodes")
    sns.regplot(x=avg_wtcs, y=avg_edge, label = "Removed Edges")
    plt.xlabel("Willingness to Compute")
    plt.ylabel("Parentless Nodes and Removed Edges")
    plt.title("Parentless Nodes and Removed Edges as a Function of WTC")
    plt.legend()
    plt.show()  

def plot_avg_trust_and_wtc(avg_trusts, wtcs):
    plt.rcParams['figure.dpi'] = 600
    sns.regplot(x=wtcs[20:], y=avg_trusts[20:])
    plt.xlabel("Willingness to Compute")
    plt.ylabel("Average Trustworthiness")
    plt.title("Average Trustworthiness as a Function of Willingness to Compute")
    plt.show()  
    
    

def plot_false_factors_and_wtc(n_false_factors, wtcs):
    plt.rcParams['figure.dpi'] = 600
    sns.regplot(x=wtcs[20:], y=n_false_factors[20:])
    plt.xlabel("Willingness to Compute")
    plt.ylabel("Number of False Factors")
    plt.title("Number of False Factors as a Function of Willingness to Compute")
    plt.show()
    
def plot_order_similarities_narrow_broad(query_factors, private_auth, avg_trusts_top):
    plt.rcParams['figure.dpi'] = 600
    
    private_auth_cumu = dict()
    private_auth_count = dict()
    avg_trust_top_cumu = dict()
    avg_trust_top_count = dict()
    
    for q, p, a in zip(query_factors, private_auth, avg_trusts_top):
        key = sum(q.keys())
        if key in private_auth_cumu:
            private_auth_cumu[key] += p
        else:
            private_auth_cumu[key] = p
            
        if key in private_auth_count:
            private_auth_count[key] += 1
        else:
            private_auth_count[key] = 1
            
        if key in avg_trust_top_cumu:
            avg_trust_top_cumu[key] += a
        else:
            avg_trust_top_cumu[key] = a
            
        if key in avg_trust_top_count:
            avg_trust_top_count[key] += 1
        else:
            avg_trust_top_count[key] = 1
            
    x = [k for k in private_auth_cumu.keys() if k < 75]
    x.sort()
    y1 = []
    y2 = []
            
    for key in x:
        
        y1.append(private_auth_cumu[key] / private_auth_count[key])
        y2.append(avg_trust_top_cumu[key] / avg_trust_top_count[key])

            
    
    fig, ax1 = plt.subplots()  # First subplot with left y-axis
    ax2 = ax1.twinx()  # Second subplot with right y-axis
    
    sns.regplot(x=x, y=y1, ax=ax1, color='g')  # Regression plot for the first y-axis
    sns.regplot(x=x, y=y2, ax=ax2, color='b')  # Regression plot for the second y-axis
    
    ax1.set_xlabel('Search Query Width (From Broad to Narrow)')
    ax1.set_ylabel('Difference HITS - Private Ranking', color='g')
    ax2.set_ylabel('Average Trustworthiness of Top HITS Nodes', color='b')
    
    ax1.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='b')
    
    plt.title("Comparing Broad to Narrow Search Queries")
    
    
    
    plt.show()
    


def plot_order_similarities(rnd, private, public):
    plt.rcParams['figure.dpi'] = 600
    size = len(rnd)
    
    avg_rnd = []
    avg_private = []
    avg_public = []
    
    step_size = 50
    for i in range(0, size - step_size - 1, step_size):
        avg_rnd.append(np.mean(rnd[i:i + step_size]))
        avg_private.append(np.mean(private[i:i + step_size]))
        avg_public.append(np.mean(public[i:i + step_size]))

        
        
    x = np.arange(len(avg_rnd))
        
    

    # Plot the data and linear regression lines
    sns.regplot(x=x, y=avg_rnd, scatter=False, label="HITS - Random Ranking")
    sns.regplot(x=x, y=avg_private, scatter=True, label="HITS - Private Factor Ranking ")
    sns.regplot(x=x, y=avg_public, scatter=False, label=" HITS - Public Ranking")
    
    plt.xlabel("Iterations")
    plt.ylabel("Normalized Ranking Difference")
    plt.title("Difference between HITS and Semantic Rankings")
    plt.legend()
    # Show the plot
    plt.show()
    
def plot_avg_false_factor_probabilities(avg_ffps):
    plt.rcParams['figure.dpi'] = 600
    size = len(avg_ffps)
    step_size = 50
    avg_avg_ffps = []
    for i in range(0, size - step_size - 1, step_size):
        avg_avg_ffps.append(np.mean(avg_ffps[i:i + step_size]))
    sns.regplot(x=np.arange(len(avg_avg_ffps)), y=avg_avg_ffps, scatter=True)
    plt.xlabel("Iterations")
    plt.ylabel("Average False Factor Probability")
    plt.title("Average False Factor Probabilities of All Nodes")
    
    plt.show()


def plot_avg_trusts(avg_trusts):
    plt.rcParams['figure.dpi'] = 600
    #sns.set_style("whitegrid")
    sns.lineplot(x=np.arange(len(avg_trusts)), y=avg_trusts)
    plt.xlabel("Iterations")
    plt.ylabel("Average Trustworthiness")
    plt.title("Average Trustworthiness of All Network Connections")
    
    plt.show()

def plot_ranking(_query_factors, _ranked_nodes, title):
    plt.rcParams['figure.dpi'] = 600
    query_factors, query_ID = arrange_factordict_data(_query_factors)
    ranked_nodes = [arrange_factordict_data(n.private_factors) for n in _ranked_nodes[:9]]
    print("len ranked", len(ranked_nodes))
        
    
    # Create the figure and subplots
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12, 10))
    
    # Remove the plot frames and ticks for all subplots
    for ax in axs.flat:
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot the first row
    sns.heatmap(query_factors, ax=axs[0, 1], vmin=0.0, vmax=1/math.log(2), cbar_kws={'label': 'Inverse Factor Size'})
    axs[0, 1].set_title('Query Factors: ' + str(_query_factors))

    # Plot the other rows
    for i in range(1, 4):
        for j in range(3):
            sns.heatmap(ranked_nodes[3*(i-1)+j][0], ax=axs[i, j], vmin=0.0, vmax=1/math.log(2))
            axs[i, j].set_title('Factors: ' + str(_ranked_nodes[3*(i-1)+j].private_factors))
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].set_frame_on(False)

    # Add a title and tight layout
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    # Show the plot
    plt.show()
    


def plot_query_and_root_set(_query_factors, _root_set, title, anti_root_set, mode):
    
    plt.rcParams['figure.dpi'] = 600
    
    query_factors, query_ID = arrange_factordict_data(_query_factors)
    
    print("mode: ", mode)
    
    
    if mode == 1:
        root_set = [arrange_factordict_data(n.public_factors) for n in _root_set[:9]]
    elif mode == 2:
        root_set = [arrange_factordict_data(n.private_factors) for n in _root_set[:9]]
    elif mode == 3:
        root_set = [arrange_factordict_data(n.public_factors) for n in anti_root_set[:9]]
    elif mode == 4:
        root_set = [arrange_factordict_data(n.private_factors) for n in anti_root_set[:9]]
        
    

    # Create the figure and subplots
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12, 10))

    # Remove the plot frames and ticks for all subplots
    for ax in axs.flat:
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot the first row
    sns.heatmap(query_factors, ax=axs[0, 1], vmin=0.0, vmax=1/math.log(2), cbar_kws={'label': 'Inverse Factor Size'})
    axs[0, 1].set_title('Query Factors: ' + str(_query_factors))

    # Plot the other rows
    for i in range(1, 4):
        for j in range(3):
            sns.heatmap(root_set[3*(i-1)+j][0], ax=axs[i, j], vmin=0.0, vmax=1/math.log(2))
            if mode == 1:
                axs[i, j].set_title('Factors: ' + str(_root_set[3*(i-1)+j].public_factors))
            elif mode == 2:
                axs[i, j].set_title('Factors: ' + str(_root_set[3*(i-1)+j].private_factors))
            elif mode == 3:
                axs[i, j].set_title('Factors: ' + str(anti_root_set[3*(i-1)+j].public_factors))
            elif mode == 4:
                axs[i, j].set_title('Factors: ' + str(anti_root_set[3*(i-1)+j].private_factors))
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    # Add a title and tight layout
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    # Show the plot
    plt.show()



def arrange_factordict_data(factordict):
    content = 1
    for factor, amount in factordict.items():
        content *= (factor ** amount)
        #content *= factor
    ID = content
    shape = (round(content ** 0.5), round(content ** 0.5))
    content = shape[0] * shape[1]
    sorted_factors = copy.deepcopy(list(factordict.keys()))
    sorted_factors.sort()

    
    
    available_space = content
    _1d = []
    while available_space > 0:
        rnd_float = 1 / random.random()
        closest = min(sorted_factors, key=lambda x: abs(x - rnd_float))
        for j in range(closest):
            _1d.append(1 / math.log(closest))
        available_space -= closest
    
    
    arr_1d = np.array(_1d[:content])
    arr_2d = arr_1d.reshape(shape)
    
    # Reverse every other row.
    for i in range(1, shape[0], 2):
        arr_2d[i] = arr_2d[i, ::-1]    
        
    return arr_2d, ID

def visualize_factordict(factordict):
    
            
    
    plt.rcParams['figure.dpi'] = 600
    arr_2d, ID = arrange_factordict_data(factordict)
    ax = sns.heatmap(arr_2d, vmin=0.0, vmax=1/math.log(2), cbar_kws={'label': 'Inverse Factor Size'})
    
    ax.set_xticks([])
    ax.set_yticks([])

    # add labels and title to the plot
    plt.title('ID: ' + str(ID) + ', Factors: ' + str(factordict))
    
    # display the plot
    plt.show() 
    
    
    
    
    
    
def heatmap_node_user_adjacency_matrix(nodes, users):
    matrix = np.zeros((len(users), len(nodes)))
    for u in users:
        for c in u.children:
            matrix[u._id][c._id] = 1
            
    plt.rcParams['figure.dpi'] = 600
            
    ax = sns.heatmap(matrix, cbar_kws={'label': 'Connection Weights'})
    
    
    # add labels and title to the plot
    plt.xlabel('Document Nodes')
    plt.ylabel('User Nodes')
    plt.title('Binary Adjacency Matrix of User-Document Connections')
    
    # display the plot
    plt.show()    
    

def draw_network_with_users(nodes, users):
    plt.rcParams['figure.dpi'] = 600
    nxGraph = G.add_users_to_nxGraph(nodes, users)
    plt.title("User-Connected Webgraph with " + str(nxGraph.number_of_nodes()) + " Nodes and " + str(nxGraph.number_of_edges()) + " Edges")
    nx.draw_networkx(nxGraph, with_labels=True)
    plt.axis('off')
    plt.show()
    

def plot_avg_vs_user_trusts(user_trusts):
    plt.rcParams['figure.dpi'] = 600
    _x = np.arange(0, len(user_trusts), 1)
    # Create a scatter plot using seaborn
    sns.scatterplot(x=_x, y=user_trusts)

    
    # Add a linear regression line using seaborn
    plt.plot(_x, [0.5 for _ in range(len(user_trusts))], label='Average Document Connection Trust')
    sns.regplot(x=_x, y=user_trusts, label='Average User Connection Trust')
    
    
    # Add title and labels
    plt.title('Average Trust of the Document Connections vs. Average Trust of User Connections')
    plt.xlabel('Iterations')
    plt.ylabel('Average Trust')
    
    plt.legend()
    
    # Show the plot
    plt.show()


def heatmap_trusts(trusts, names):
    
    plt.rcParams['figure.dpi'] = 600
    
    # create a heatmap using Seaborn
    ax = sns.heatmap(trusts, cbar_kws={'label': 'Trust Values'})
    
    ax.set_yticklabels(names, rotation=0)
    
    # add labels and title to the plot
    plt.xlabel('Nodes')
    plt.title('Trust Comparison')
    
    # display the plot
    plt.show()
    
def evolution_mean_order_similarities(similarities, params):
    plt.rcParams['figure.dpi'] = 600
    
    x = range(len(similarities[0]))
    
    for s, param in zip(similarities, params):
        if param[5] == "Standard HITS":
            sns.lineplot(x=x, y=s, label="Random Order")
        else:   
            sns.lineplot(x=x, y=s, label=param[5])
    
    plt.xlabel('Iterations')
    plt.ylabel('Mean Order Difference')
    plt.title('Mean Order Differences Compared to Standard HITS')
    
    # Display the legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()

def heatmap_trust(nodes, trust_IDs):
    
    plt.rcParams['figure.dpi'] = 600
    y = [nodes[_id].trust for _id in trust_IDs]
    x = [str(_id) for _id in trust_IDs]
    
    sns.scatterplot(x=x, y=y)
    
    plt.xlabel('Node ID')
    plt.ylabel('Trust Values')
    plt.title('Distribution of Trust Values Among Nodes')
    
    plt.show()
    
    
def heatmap_adjacency_matrix(nodes):
    matrix = np.zeros((len(nodes), len(nodes)))
    for n in nodes:
        for k, v in n.edges.items():
            matrix[n._id][k] = v
            
    plt.rcParams['figure.dpi'] = 600
            
    ax = sns.heatmap(matrix, cbar_kws={'label': 'Edge Weights'})
    
    
    # add labels and title to the plot
    plt.xlabel('Children')
    plt.ylabel('Parents')
    plt.title('Adjacency Matrix')
    
    # display the plot
    plt.show()
        


def heatmap_auth_rankings(auths, names):
    plt.rcParams['figure.dpi'] = 600
    # create a heatmap using Seaborn
    ax = sns.heatmap(auths, annot=True, cbar_kws={'ticks':np.arange(0, len(auths[0]), 1),'label': 'Node IDs'})
    
    ax.set_yticklabels(names, rotation=0)
    
    # add labels and title to the plot
    plt.xlabel('Authority Rank')
    plt.title('Authority Value Ranking from Lowest to Highest')
    
    # display the plot
    plt.show()
    
    
def heatmap_hub_rankings(hubs, names):
    plt.rcParams['figure.dpi'] = 600
    # create a heatmap using Seaborn
    ax = sns.heatmap(hubs, annot=True, cbar_kws={'ticks':np.arange(0, len(hubs[0]), 1),'label': 'Node IDs'})
    
    ax.set_yticklabels(names, rotation=0)
    
    # add labels and title to the plot
    plt.xlabel('Hub Rank')
    plt.title('Hub Value Ranking from Lowest to Highest')
    
    # display the plot
    plt.show()
    
def plot_auth_distribution_transitions(auth_distributions, names):
    
    plt.rcParams['figure.dpi'] = 600
    
    # Compute intermediate distributions
    distributions = auth_distributions
    
    # Plot the transition using histograms
    fig, axes = plt.subplots(len(names), 5, figsize=(15, len(names) * 3))
    
    row_titles = names
    
    for row, intermediate_distributions in enumerate(distributions):
        for col, distribution in enumerate(intermediate_distributions):
            sns.histplot(distribution, kde=True, ax=axes[row, col], color='skyblue')
            axes[row, col].set_title(f'Iteration {col*5}/{20}')
            axes[row, col].set_xlabel('')
            axes[row, col].set_ylabel('')
            
            #axes[row, col].relim()
            if row == 2 or row == 3 or row == 4:
                #axes[row, col].set_xlim(0, 0.05)
                pass
                
            #axes[row, col].set_xlim(-5, 10)  # Adjust x-axis limits if needed
    
        axes[row, 0].set_ylabel(row_titles[row])
    

    
    plt.tight_layout()
    plt.show()
    
    
    
if __name__ == "__main__":
    
    plot_FFP_distribution([0.01, 0.5, 0.39, 0.999])

    
    
 
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





