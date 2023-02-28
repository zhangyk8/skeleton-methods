# -*- coding: utf-8 -*-

# Author: Zeyu Wei, Yikun Zhang
# Last Editing: February 27, 2022

# Description: Skeleton Construction

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
from scipy.stats import gaussian_kde
import collections

#================================================================================#

def constructKnots(X, centers = None, labels = None, k = None, rep = 100):
    """
      construct knots using overfitting kMeans
  
      Parameters
      ----------
      X : np.array
          the data ndarray
      centers: np.array of the knots, can be provided
      labels: np.array of the knot label that each data point belongs to
      k: the number of knots
      rep: times of random initialization for kMeans
  
      Returns
      -------
      centers: ndarray of the constructed knots
      cluster: array of knot labels to which the data points belong to
      nknots: number of knots
      withinss: within cluster sum of squares for each Voronoi cell corresponding to the knots
    """
    n, d = X.shape
    #construct knots
    if centers is None and labels is None:
        # Overfitting k-means
        #setting the number of knots k
        if k is None:
            k = round(np.sqrt(n))
    
        km = KMeans(n_clusters = k, n_init = rep)
        km.fit(X)
        centers = km.cluster_centers_
        labels = km.labels_
    
    elif labels is None:#centers provided but not labels
        nbrs = NearestNeighbors(n_neighbors=1).fit(centers)
        labels = nbrs.kneighbors(X, return_distance=False)
        k = len(centers)
          
    elif centers is None:#labels provided but not centers
        elements_count = collections.Counter(labels)
        k = len(elements_count.items())
        centers = np.array([[0.0 for col in range(d)] for row in range(k)])
        for key, value in elements_count.items():
            centers[key,] = np.mean(X[labels == key,], axis=0)
      
    else:
        k = len(centers)
        
      
    withinss = np.array([0.0]*k)
    for i in range(k):
        withinss[i] = np.sum((X[labels == i,]-centers[i,])**2)
      
    return {"centers":centers, "cluster":labels, "nknots":k, "withinss": withinss}


def skelWeights(X, conKnots, wedge= "all", h = "silverman", R0 = None, idx_frustum = False):
    """
      calculate weights based on constructed skeleton
  
      Parameters
      ----------
      X : np.array of the data ndarray
      conKnots: list object returned by constructKnots
      wedge: which edge measure to include
      h: a number or string for the bandwidth for KDE used for Face and Tube density
      R0: disk radius for Tube density
      idx_frustum: whether setting same disk radius for all edges
  
      Returns
      -------
      nn: indices of the 2 nearest knots for each data point
      voron_weights: voronoi density weights between the knots
      face_weights: face density weights between the knots
      frustum_weights: Tube density weights between the knots. Called frustum as for different radii for different knots.
      R: radius used for Tube density
      avedist_weights: average distance density weights between the knots
    """
    m = conKnots["nknots"]
    knots = conKnots["centers"]
    knotlabels = conKnots["cluster"]
    withinss = conKnots["withinss"]
  
    #identify what edge weights to include
    edge_include = [False]*4
    if wedge == 'all':
        edge_include = [True]*4
    else:
        edge_include = np.isin(['voronoi', 'face', 'frustum', 'avedist'], wedge )
    
    #edge weight matrices
    if edge_include[0]:
        voron_weights = np.array([[0.0 for col in range(m)] for row in range(m)])
    if edge_include[1]:
        face_weights = np.array([[0.0 for col in range(m)] for row in range(m)])
    if edge_include[3]:
        avedist_weights = np.array([[0.0 for col in range(m)] for row in range(m)])
    if edge_include[2]:
        frustum_weights = np.array([[0.0 for col in range(m)] for row in range(m)])
        ### setting the threshold for each disk
        if R0 is None: #fill in average within cluster variance if not specified
            elements_count = collections.Counter(knotlabels)
            knotsizes = np.array([0]*m)
            for key, size in elements_count.items():
                knotsizes[key] = size
            R_cluster = np.sqrt(withinss/(knotsizes-1))
            R0 = np.mean(R_cluster)
            if idx_frustum: #same radius for all knots
                R0_lv = np.array([R0]*m)
            else: #different radius for each knot
                R0_lv = R_cluster
        elif len(R0)== 1: # only one specified R0
            R0_lv = np.array([R0]*m)
            idx_frustum = True
        elif len(R0) == m:
            R0_lv = R0
            idx_frustum = False
        else:
            elements_count = collections.Counter(knotlabels)
            knotsizes = np.array([0]*m)
            for key, size in elements_count.items():
                knotsizes[key] = size
            R_cluster = np.sqrt(withinss/(knotsizes-1))
            R0 = np.mean(R_cluster)
            if idx_frustum: #same radius for all knots
                R0_lv = np.array([R0]*m)
            else: #different radius for each knot
                R0_lv = R_cluster
  
  
  
    # calculate 2 Nearest Neighbor Indices
    nbrs = NearestNeighbors(n_neighbors=2).fit(knots)
    X_nn = nbrs.kneighbors(X, return_distance=False) #2-nearest neighbor calculation
  
    for i in range(m-1): #loop through knots pairs
        center1 = knots[i]
        wi1 = np.where(X_nn[:,0] == i)[0]
        wi2 = np.where(X_nn[:,1] == i)[0]
        for j in range(i+1,m):
            center2 = knots[j]
            wj1 = np.where(X_nn[:,0] == j)[0]
            wj2 = np.where(X_nn[:,1] == j)[0]
            #data point indices within 2nn neighborhood of knots i, j
            nn2ij = np.union1d(np.intersect1d(wi1, wj2), np.intersect1d(wi2, wj1))
        
            if len(nn2ij) < 1 :#not in Delaunnay Triangulation
                if edge_include[0]:
                    voron_weights[i,j] = 0.0
                if edge_include[1]:
                    face_weights[i,j] = 0.0
                if edge_include[2]:
                    frustum_weights[i,j] = 0.0
                if edge_include[3]:
                    avedist_weights[i,j] = 0.0
                continue
        
        # Euclidean distance between two centers
        d12 = np.sqrt(sum((center1-center2)**2))
  
        #compute the Voronoi density 
        if edge_include[0]: 
            voron_weights[i,j] = len(nn2ij)/d12
        
        #compute the face density
        if edge_include[1]:
            v0 = (center2-center1)/d12 #direction vector
            X_ij = X[np.union1d(wi1, wj1)]
            p0_length = np.dot(X_ij-(center1+center2)/2,v0 )  #projected distance to middle point of the edge
            p_dot = p0_length/d12 #standardize the projected distances into proportions
            
            kde = gaussian_kde(p_dot, bw_method=h) #KDE with projected points
            face_weights[i,j] = kde.evaluate([0]) #interpolated density at middle point
            #end face density calculation
        
        #compute tube density
        if edge_include[2]: 
            if not edge_include[1]: # recompute some quantities in Face density calculation
                v0 = (center2-center1)/d12 #direction vector
                X_ij = X[np.union1d(wi1, wj1)]
    
            p1_length = np.dot(X_ij-center1,v0 )  # length to center1 after projection
            p1_dot = p1_length/d12 #standardize the projected distances into proportions
            perp_length = np.sqrt(np.sum((X_ij -center1)**2, axis=1)-p1_length**2) #orthogonal distance to the center-passing line
        
            #threshold for each datapoint
            R0_threshold = R0_lv[i] + (R0_lv[j]-R0_lv[i])*p1_dot
            w_edge = np.where(perp_length<R0_threshold)#points that can be used in KDE
            
            if len(w_edge)>1: #KDE works with more than 1 data point
                kde = gaussian_kde(p1_dot[w_edge], bw_method=h) #KDE with projected points
                # finding the minimal density
                frustum_weights[i,j] = min(kde.evaluate(np.linspace(0.0, 1.0, num=100)))
        #end frustum density calculation
  
        # compute avedist density
        if edge_include[3]: 
            dists = distance_matrix(X[wi1,], X[wj1,])
            avedist_weights[i,j] = np.mean(dists)
        #end avedist density
  
    output = {"nn": X_nn}
    if edge_include[0]:
        output.update( {"voron_weights": voron_weights + np.transpose((voron_weights))})
    if edge_include[1]:
        output.update( {"face_weights": face_weights + np.transpose((face_weights))})
    if edge_include[2]:
        output.update( {"frustum_weights": face_weights + np.transpose((frustum_weights))})
        output.update( {"R": R0_lv})
    if edge_include[3]:
        output.update( {"avedist_weights": avedist_weights })
    
    return(output)


def Skeleton_Construction(X, centers = None, labels = None, k = None, rep = 100, 
                          wedge= "all", h = "silverman", R0 = None, idx_frustum = False):

    conKnots = constructKnots(X, centers, labels, k, rep = 100)
    output = conKnots
    if wedge is not None:
        edgeWeights = skelWeights(X, conKnots, wedge, h, R0, idx_frustum)
        output.update(edgeWeights)
    return(output)


def skelProject(X, skeleton, nn = None):
    """
      Project data points onto the skeleton
  
      Parameters
      ----------
      X: the matrix of the data points to be projected
      skeleton: a list returned by segment_Skeleton()
      nn: the matrix where each row records the indices of the two closest centers of the corresponding data point.
  
  
      Returns
      -------
      A vector where each entry records the projection proportion of the corresponding data point 
      on the edge by the two closest knots, from the smaller index knot. NA if the two closest knots not connected.
  
    """
    knots = skeleton["centers"]
    kkdists = skeleton["kkdists"]
    g = skeleton["g"]
    n = len(X)
  
    if nn is None:
        nbrs = NearestNeighbors(n_neighbors=2).fit(knots)
        nn = nbrs.kneighbors(X, return_distance=False) #2-nearest neighbor calculation
  
    #calculate the projection for each data point
    #a proportion between [0,1] is recorded
    px = [None]*n
    for i in range(n):
        #begin calculate the projection for each data point
        if g.are_connected(nn[i,0], nn[i,1]):
            #if the two nearest knots are connected
            k0 = np.min(nn[i,:])
            k1 = np.max(nn[i,:])
            #directional vector
            #always use the smaller index as the starting point
            v = (knots[k1,:]-knots[k0,:])/kkdists[k0,k1]
            prop = np.dot(X[i,:]-knots[k0,], v)/kkdists[k0,k1] #projected proportion
            px[i] = max(min(prop, 1),0) #confine to [0,1]
      #leave the projection to knot cases as None
    
    return(px)


def dskeleton(nnc1, nnc2, px1, px2, skeleton):
    g = skeleton["g"]
    kkdists = skeleton["kkdists"]
  
    #simplify calculation by only focusing on pair of points sharing at least one knot
    if len(np.intersect1d(nnc1,nnc2))<1:
        return(float('inf'))
    #case when both data points cannot be projected onto an edge
    elif px1 is None and px2 is None:
        # g.shortest_paths(nnc1[0], nnc2[0], weights='weight') #for general graph distance
        return(kkdists[nnc1[0], nnc2[0]])
    #case when point id1 can be projected onto an edge but not point 2
    elif px1 is not None and px2 is None:
        btnodes1 = g.shortest_paths(nnc1[0], nnc2[0], weights='weight')[0][0]
        btnodes2 = g.shortest_paths(nnc1[1], nnc2[0], weights='weight')[0][0]
        if not np.isfinite(btnodes1) and not np.isfinite(btnodes2):
            return(float('inf'))
        elif btnodes1 < btnodes2:#shortest path from nnc1[0]
            if nnc1[0] < nnc1[1]: #nnc1[0] is the reference point for projection
                return(btnodes1 + kkdists[nnc1[0], nnc1[1]]*px1 )
            else:
                return(btnodes1 + kkdists[nnc1[0], nnc1[1]]*(1-px1) )
        else: ##shortest path from nnc1[1], btnodes2 < btnodes1
            if nnc1[0] < nnc1[1]: #nnc1[0] is the reference point for projection
                return(btnodes2 + kkdists[nnc1[0], nnc1[1]]*(1-px1) )
            else:
                return(btnodes2+ kkdists[nnc1[0], nnc1[1]]*px1 )
    #case when point id2 can be projected onto an edge
    elif px1 is None and px2 is not None:
        btnodes1 = g.shortest_paths(nnc1[0], nnc2[0], weights='weight')[0][0]
        btnodes2 = g.shortest_paths(nnc1[0], nnc2[1], weights='weight')[0][0]
        if not np.isfinite(btnodes1) and not np.isfinite(btnodes2):
            return(float('inf'))
        elif btnodes1 < btnodes2:#shortest path from nnc2[0]
            if nnc2[0] < nnc2[1]: #nnc2[0] is the reference point for projection
                return(btnodes1 + kkdists[nnc2[0], nnc2[1]]*px2 )
            else:
                return(btnodes1 + kkdists[nnc2[0], nnc2[1]]*(1-px2) )
        else: ##shortest path from nnc2[1], btnodes2 < btnodes1
            if nnc2[0] < nnc2[1]: #nnc2[0] is the reference point for projection
                return(btnodes2 + kkdists[nnc2[0], nnc2[1]]*(1-px2) )
            else:
                return(btnodes2+ kkdists[nnc2[0], nnc2[1]]*px2 )
    else: #case when both points are projected onto edges
        btnodes = [g.shortest_paths(nnc1[0], nnc2[0], weights='weight')[0][0],
        g.shortest_paths(nnc1[0], nnc2[1], weights='weight')[0][0],
        g.shortest_paths(nnc1[1], nnc2[0], weights='weight')[0][0],
        g.shortest_paths(nnc1[1], nnc2[1], weights='weight')[0][0]
        ]
        if np.all(np.invert(np.isfinite(btnodes))): #all are infinite
            return(float('inf'))
    
        idmin = np.argmin(btnodes)
        temp = np.min(btnodes)
        if idmin == 0: #nnc1[0] and nnc2[0] are the closest
            if nnc1[0] < nnc1[1]:#nnc1[0] is the reference point for id1 projection
                temp = temp + kkdists[nnc1[0], nnc1[1]]*px1
            else:
                temp = temp + kkdists[nnc1[0], nnc1[1]]*(1-px1)
            if nnc2[0] < nnc2[1]: #nnc2[0] is the reference point for id2 projection
                temp = temp + kkdists[nnc2[0], nnc2[1]]*px2
            else:
                temp = temp + kkdists[nnc2[0], nnc2[1]]*(1-px2)
            return(temp)
        elif idmin == 1: #nnc1[0] and nnc2[1] are closest
            if nnc1[0] < nnc1[1]:#nnc1[0] is the reference point for id1 projection
                temp = temp + kkdists[nnc1[0], nnc1[1]]*px1
            else:
                temp = temp + kkdists[nnc1[0], nnc1[1]]*(1-px1)
            if nnc2[0] < nnc2[1]: #nnc2[0] is the reference point for id2 projection
                temp = temp + kkdists[nnc2[0], nnc2[1]]*(1-px2)
            else:
                temp = temp + kkdists[nnc2[0], nnc2[1]]*px2
            return(temp)
        elif idmin == 2: #nnc1[1] and nnc2[0] are closest
            if nnc1[0] < nnc1[1]:#nnc1[0] is the reference point for id1 projection
                temp = temp + kkdists[nnc1[0], nnc1[1]]*(1-px1)
            else:
                temp = temp + kkdists[nnc1[0], nnc1[1]]*px1
            if nnc2[0] < nnc2[1]:#nnc2[1] is the reference point for id2 projection
                temp = temp + kkdists[nnc2[0], nnc2[1]]*px2
            else:
                temp = temp + kkdists[nnc2[0], nnc2[1]]*(1-px2)
            return(temp)
        else: #nnc1[2] and nnc2[2] are closest
            if nnc1[0] < nnc1[1]: #nnc1[1] is the reference point for id1 projection
                temp = temp + kkdists[nnc1[0], nnc1[1]]*(1-px1)
            else:
                temp = temp + kkdists[nnc1[0], nnc1[1]]*px1
            if nnc2[0] < nnc2[1]: #nnc2[0] is the reference point for id2 projection
                temp = temp + kkdists[nnc2[0], nnc2[1]]*(1-px2)
            else:
                temp = temp + kkdists[nnc2[0], nnc2[1]]*px2
            return(temp)
        
        
# pairwise skeleton-based distance
def pairdskeleton(xnn, ynn, px,  py, skeleton):
    """
      pairwise skeleton-based distance
  
      Parameters
      ----------
      xnn/ ynn: the matrix where each row records the indices of the two closest centers of the corresponding data point.
      px/ py: projecting proportions as returned by skelProject()
      skeleton: a list returned by segment_Skeleton()
  
  
      Returns
      -------
      pairwise skeleton-based distances between the two sets of points on the skeleton.
  
    """
    xydists = np.array([[float('inf') for col in range(len(py))] for row in range(len(px))])
    for i in range(len(px)):
      #calculate graph distances between knots and sample points
      for j in range(len(py)):
        xydists[i,j] = dskeleton(xnn[i,], ynn[j,],  px[i], py[j], skeleton = skeleton)
  
    return(np.array(xydists))

