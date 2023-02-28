# -*- coding: utf-8 -*-

# Author: Zeyu Wei, Yikun Zhang
# Last Editing: February 27, 2022

# Description: This script contains the utility functions for using our package 
# in practice.

import numpy as np

#================================================================================#

def distance_mat(A, B, squared=False):
    """
    Compute all pairwise distances between vectors in A and B.

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.
    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared


def twoMoon(numObjects = 180, shape1a = -0.4, shape2b = 1, shape1rFrom = 0.8, shape1rTo = 1.2, shape2rFrom = 0.8, shape2rTo = 1.2):
    """
    Generating twoMoon data

    Parameters
    ----------
    
    Returns
    -------
    """
    nrow = numObjects*2
    x =  np.array([[0.0 for col in range(2) ] for row in range(nrow)])
    for i in range(nrow):
        alpha = np.random.uniform(low=0.0, high=2*np.pi) 
        if i >= numObjects:
            r = np.random.uniform(shape2rFrom, shape2rTo)
        else:
            r = np.random.uniform(shape1rFrom, shape1rTo)
        tmp1 = r * np.cos(alpha)
        tmp2 = r * np.sin(alpha)
        if i < numObjects:
            x[i, 0] = shape1a + abs(tmp1)
            x[i, 1] = tmp2
        else:
            x[i, 0] = -abs(tmp1)
            x[i, 1] = tmp2 - shape2b
    
    label = np.array([0 for row in range(numObjects)]+ [1 for row in range(numObjects)])
    return {"data": x, "label" :label}


def Yinyang_data(n_m=400,n_c=200,n_r=2000,var_c=0.01,sd_r=0.1, d=2, sd_high=0.1):
    X_m = twoMoon(numObjects = n_m)
    
    X1= np.random.multivariate_normal([0.5, -1.5], [[var_c, 0.0], [0.0, var_c]], n_c)
    X2= np.random.multivariate_normal([-1, 0.5], [[var_c, 0.0], [0.0, var_c]], n_c)

    th = np.random.uniform(0, 2*np.pi, n_r)
    X31 = 2.5*np.cos(th) - 0.25 + np.random.normal(0, sd_r, n_r)
    X32 = 2.5*np.sin(th) -0.5 + np.random.normal(0, sd_r, n_r)

  
    X = np.vstack((X_m["data"], X1, X2, np.stack((X31, X32), axis=-1)))
    if d>2:
      noised = np.random.normal(0, sd_high, size = (len(X), int(d-2)))
      X = np.hstack((X, noised))

    label = np.concatenate((X_m["label"], [2 for _ in range(n_c)] ,  [3 for _ in range(n_c)] , [4 for _ in range(n_r)] ), axis=None)
    
    return {"data": X, "label" :label}