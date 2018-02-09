import numpy as np
import networkx as nx

def getNetwork(W, W_fixed):
    """
    Turns a numpy matrix into a networkX directed graph object

    Input:  Two numpy matrices. One with the expert weights and one with
            the fixed weights
    Returns:A networkX directed graph object
    """

    W = np.matrix(W + W_fixed)
    W[W!=0] = 1
    W = np.array(W, dtype=int)
    return nx.DiGraph(W)

def getSimilarity(W_expert, W_naive, W_fixed):
    """
    Builds a similarity measure between the expert model and a naive model.
    For every causal relationship in the original model, the pathlength of
    this relationship in the naive model is calculated.
    The similarity measure is a sum over the inverse of these pathlengths.

    Input:  Two numpy matrices. One with the expert weights and one with
            the fixed weights
    Returns:A float [0,1], similarity measure
    """

    G_expert = getNetwork(W_expert, W_fixed)
    G_naive = getNetwork(W_naive, W_fixed)

    sim, tot = 0, 0
    for (_, p_e), (_, p_n) in zip(
        nx.shortest_path_length(G_expert),
        nx.shortest_path_length(G_naive)
    ):
        for k in p_e.keys():
            if p_e[k] == 1:
                tot += 1
                try:
                    sim += 1/p_n[k]
                except: continue
    return sim/tot
