# -*- coding: utf-8 -*-

# Author: Zeyu Wei, Yikun Zhang
# Last Editing: February 27, 2022

# Description: Skeleton Clustering

import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score
from scipy.spatial import distance_matrix
import igraph

#================================================================================#

def cluster_weights(weights, Xknotcluster, kcut, Y=None, method="single"):
    """
      perform clustering using the skeleton-based weights
  
      Parameters
      ----------
      weights : ndarray of the edge weights to cluster based on
      Xknotcluster: array of knot labels where each data point belongs to
      kcut: the number of resulting disconnected components
      Y: true class labels. Used to calculate performance
      method: linkage method for hierarchical clustering
  
      Returns
      -------
      a list with the clustering results
    """
    p_dist = np.max(weights)+1 - weights
    np.fill_diagonal(p_dist, 0.0)
    condense_dist= squareform(p_dist, checks=False)
    hclust = linkage(condense_dist, method)
    knot_labs = fcluster(hclust, kcut, criterion='maxclust') # membership for knots
    X_labs = np.array(knot_labs)[np.array(Xknotcluster)] # labels for each data point
    if Y is None:
        adj_rand = None
    else:
        adj_rand = adjusted_rand_score(X_labs, Y)
    return({"hclust":hclust, "knot_labs":knot_labs, "X_labs":X_labs, "adj_rand":adj_rand})


def segment_Skeleton(skeleton,  wedge = "voron_weights", kcut=1, cut_method = "single", tol = 1e-10):
    """
      segment the skeleton graph of a dataset based on edge weights
  
      Parameters
      ----------
      skeleton: obejct resulting from Skeleton_Construction()
      wedge: which edge weights used to segment skeleton
      kcut: the number of resulting disconnected components
      cut_method: linkage method for hierarchical clustering
      tol: tolerance for float error
  
      Returns
      -------
      Adding items to the skeleton:
      kkdists: Euclidean distances between knots
      cutWeights: The weight matrix with values for cut edges set to 0
      g: the igraph object of the segmented skeleton
  
    """
    knots = skeleton["centers"]
    X_knotlabels = skeleton["cluster"]
  
    #similarities
    weights = skeleton[wedge]
    hclustKnot = cluster_weights(weights, X_knotlabels, kcut = kcut, method=cut_method)
  
  
    #choose correct cut
    cutheight = hclustKnot["hclust"][-kcut][2]
    #cut some edges
    weights[weights<(np.max(weights)-cutheight+tol)] = 0
  
    #Euclidean distance between centers
    kkdists = distance_matrix(knots,knots)
    kkdists[weights == 0] = 0
  
    # ###############################################
    # # get the graph based on Euclidean similarities
    # ###############################################
    g = igraph.Graph.Weighted_Adjacency(kkdists.tolist(),  mode='undirected')
    # layout = g.layout("coords_fr")
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # layout = g.layout("kk")
    # igraph.plot(g, layout=layout, target=ax)
    skeleton.update({"kkdists": kkdists, "cutWeights": weights, "g" :g}) 
    return(skeleton)