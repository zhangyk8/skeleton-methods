# -*- coding: utf-8 -*-

# Author: Zeyu Wei, Yikun Zhang
# Last Editing: February 27, 2022

# Description: Skeleton Regression

import numpy as np
from skeleton import Skeleton_Construction, skelProject, pairdskeleton
from clustering import segment_Skeleton
import scipy
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

#================================================================================#

def hns(x, deriv = 0):
    """
      compute the normal scale kernel bandwidth
  
      Parameters
      ----------
      x: array of values
      deriv: derivative orders used for calculation
  
  
      Returns
      -------
      the normal scaling bandwidth
  
    """
    n = len(x)
    if n < 1:
        return np.nan
    d = 1
    r = deriv
    h = (4/(n * (d + 2 * r + 2)))**(1/(d + 2 * r + 4)) * np.std(x)
    return(h)


def skelBand(xkdists):
    """
      compute the bandwidth for skeleton kernel regression
  
      Parameters
      ----------
      xkdists: skeleton-based distances between the datapoints and the knots
  
      Returns
      -------
      the bandwidth
  
    """
    k = len(xkdists)
    knotbands = [0.0 for _ in range(k)]
    for i in range(k):
        knotbands[i] = hns(xkdists[i,np.isfinite(xkdists[i,])])
  
    return(np.nanmean(knotbands))


def epanechnikov_kernel(x,width):
    """
    For a point x_0 in x, return the weight for the given width.
    """
    def D(t):
        if t <= 1:
            #return 3/4*float(1-t*t) <== why doesn't this work?
            return float(1-t*t)*3/4
        else:
            return 0
    return D(abs(x)/width)

def tri_cube_kernel(x,width):
    def D(t):
        if t <= 1:
            return float(1-t*t*t)**3
        else:
            return 0
    return D(abs(x)/width)


def kernel_smoother(X,Y,kernel,bandwidth):
    """
    Generalization with custom kernel.
    
    Parameters:
        X: the vector of feature data
        x_0: a particular point in the feature space
        kernel: kernel function
        width: kernel width
    
    Return value:
        The estimated regression function at x_0.
    """
    kernel_weights = [kernel(x,bandwidth) for x in X]
    
    weighted_average = np.average(Y,weights=kernel_weights)
    return weighted_average


def skelKernel(trainX, trainY,  testX, skeleton = None, hreg= None, testtraindists = None,
centers = None, labels = None, nknot = None, rep = 100, wedge = "voronoi", hden = "silverman", R0 = None, idx_frustum = False, #arguments for skeleton construction
wedge_cut = "voron_weights", kcut=1, cut_method = "single", #parameters to cut skeleton
tol = 1e-10
):
    """
    Generalization with custom kernel.
    
    Parameters:
        trainX: training covariates
        trainY: training response
        testX: test covariates
        skeleton: object resulting from segment_Skeleton()
        hreg: bandwidth for kernel smoothing regression
        testtraindists: matrix of skeleton-based distances between test and training data
    Parameters for Skeleton_Construction
    Parameters for segment_Skeleton
  
    
    Return value:
        The estimated regression values for test data.
    """
    if testtraindists is None:
        if skeleton is None:
            skeleton = Skeleton_Construction(trainX, centers = centers,labels = labels, k = nknot,  rep = rep, 
                                             wedge= wedge, h = hden, R0 = R0, idx_frustum = idx_frustum)
            skeleton = segment_Skeleton(skeleton, wedge = wedge_cut, kcut=kcut, cut_method = cut_method)
    
        trainnn = skeleton["nn"]
        trainpx = skelProject(trainX, skeleton, nn = trainnn)
    
        nbrs = NearestNeighbors(n_neighbors=2).fit(skeleton["centers"])
        testnn = nbrs.kneighbors(testX, return_distance=False) #2-nearest neighbor calculation
        testpx=skelProject(testX, skeleton, nn = testnn)
    
        testtraindists = pairdskeleton(testnn,trainnn, testpx, trainpx,  skeleton)
      
  
    if hreg is None:
        nbrs = NearestNeighbors(n_neighbors=2).fit(skeleton["centers"])
        knotnn = nbrs.kneighbors(skeleton["centers"], return_distance=False) #2-nearest neighbor calculation
        knotpx=skelProject(skeleton["centers"], skeleton, nn = knotnn)
        try:
            trainpx
        except NameError: # calculate projection for trainX
            trainpx = skelProject(trainX, skeleton, nn = skeleton["nn"])
        xkdists = pairdskeleton( knotnn,skeleton["nn"],  knotpx, trainpx,  skeleton)
        hreg = skelBand(xkdists)
        if hreg < np.quantile(xkdists[np.isfinite(xkdists)], 0.1) or hreg > np.quantile(xkdists[np.isfinite(xkdists)], 0.5):
            hreg = np.quantile(xkdists[np.isfinite(xkdists)], 0.25)
  
    ntest =   len(testX)
    testfit = [0.0 for _ in range(ntest)]
    trainY = np.array(trainY)
    #predictions on test data
    for i in range(ntest):
        z = testtraindists[i,]
        useid = np.isfinite(z)
        z = z[useid]
        if len(z)==1: #happens when there is only 1 finite distance training point
            testfit[i] = trainY[useid][0]
        elif np.std(z) < tol: # case when all x are close, which happens on singly separated knot
            testfit[i] = np.mean(trainY[useid])
        else:
            kernel_weights = [scipy.stats.norm.pdf(val/hreg) for val in z]
            testfit[i] = np.average(trainY[useid],weights=kernel_weights)
  
    return(testfit)


def skelLinear(newnn, newpx, trainnn, trainpx, trainY, skeleton):
    numknots = skeleton["nknots"]
    ntrain = len(trainY)
    ntest = len(newpx)
  
    #modified data matrix for training data based projected values
    trainZ = np.array([[0.0  for col in range(numknots)] for row in range(ntrain) ])
    for i in range(ntrain):
        if trainpx[i] is None: #where the data point is projected to nearest knot
            trainZ[i,trainnn[i,0]] = 1
        else:
            k0 = np.min(trainnn[i,])
            k1 = np.max(trainnn[i,])
            #Y0 + px(Y1-Y0), hence px(Y1), (1-px)Y0
            trainZ[i, k0] = 1-trainpx[i] #*edgedists[k0,k1]
            trainZ[i, k1] = trainpx[i]
  
    #modified data matrix for test data based projected values
    testZ = np.array([[0.0  for col in range(numknots)] for row in range(ntest)] )
    for i in range(ntest):
        if newpx[i] is None: #where the data point is projected to nearest knot
            testZ[i,newnn[i,0]] = 1
        else:
            k0 = np.min(newnn[i,])
            k1 = np.max(newnn[i,])
            #Y0 + px(Y1-Y0), hence px(Y1), (1-px)Y0
            testZ[i, k0] = 1-newpx[i] #*edgedists[k0,k1]
            testZ[i, k1] = newpx[i]
  
    lmod = LinearRegression(fit_intercept= False).fit(trainZ, trainY)
    lmpred = lmod.predict(testZ)
  
    return({"trainZ": trainZ, "testZ":testZ, "model":lmod, "pred":lmpred})


def skelLspline(trainX, trainY, testX, 
testnn = None, testpx = None, trainnn = None, trainpx=None, skeleton = None):
    """
    S-Lspline regression from raw data
    
    Parameters:
        trainX: training covariates
        trainY: training response
        testX: test covariates
        testnn/trainnn: indices of two nearest knots for test/train covariates
        testpx/trainpx: projection proportions for test/train covariates
        skeleton: object resulting from segment_Skeleton()
    
    Return value:
        The estimated regression results:
        trainZ: transformed training covariates
        testZ: transformed test covariates
        model: the regression model return from sklearn.linear_model.LinearRegression
        pred: predicted values for test data
  
    """
    if skeleton is None:
        skeleton = Skeleton_Construction(trainX)
        skeleton = segment_Skeleton(skeleton)
    if trainnn is None:
        trainnn = skeleton["nn"]
    if trainpx is None:
        trainpx = skelProject(trainX, skeleton, nn = trainnn)
    if testnn is None:
        nbrs = NearestNeighbors(n_neighbors=2).fit(skeleton["centers"])
        testnn = nbrs.kneighbors(testX, return_distance=False) #2-nearest neighbor calculation
    if testpx is None:
        testpx=skelProject(testX, skeleton, nn = testnn)
    
    return skelLinear(testnn, testpx, trainnn, trainpx, trainY, skeleton)


def skelQuadratic(newnn, newpx, trainnn, trainpx, trainY, skeleton):
    numknots = skeleton["nknots"]
    ntrain = len(trainY)
    ntest = len(newpx)
  
    #modified data matrix for training data based projected values
    trainZ = np.array([[0.0  for col in range(2*numknots)] for row in range(ntrain) ])
    for i in range(ntrain):
        if trainpx[i] is None: #where the data point is projected to nearest knot
            trainZ[i,trainnn[i,0]] = 1
        else:
            k0 = np.min(trainnn[i,])
            k1 = np.max(trainnn[i,])
            trainZ[i, k0] = 2*trainpx[i]**3 - 3*trainpx[i]**2 + 1
            trainZ[i, k1] = -2*trainpx[i]**3 + 3*trainpx[i]**2
            trainZ[i, (k0+numknots)] = trainpx[i]**3 - 2*trainpx[i]**2 + trainpx[i]
            trainZ[i, (k1+numknots)] = trainpx[i]**3 - trainpx[i]**2
  
    #modified data matrix for test data based projected values
    testZ = np.array([[0.0  for col in range(2*numknots)] for row in range(ntest)] )
    for i in range(ntest):
        if newpx[i] is None: #where the data point is projected to nearest knot
            testZ[i,newnn[i,0]] = 1
        else:
            k0 = np.min(newnn[i,])
            k1 = np.max(newnn[i,])
            #Y0 + px(Y1-Y0), hence px(Y1), (1-px)Y0
            testZ[i, k0] = 2*(newpx[i]**3) - 3*(newpx[i]**2) + 1
            testZ[i, k1] = -2*(newpx[i]**3) + 3*(newpx[i]**2)
            testZ[i, (k0+numknots)] = newpx[i]**3 - 2*(newpx[i]**2) + newpx[i]
            testZ[i, (k1+numknots)] = newpx[i]**3 - newpx[i]**2
  
    lmod = LinearRegression(fit_intercept= False).fit(trainZ, trainY)
    lmpred = lmod.predict(testZ)
  
    return({"trainZ": trainZ, "testZ":testZ, "model":lmod, "pred":lmpred})


def skelCubic(newnn, newpx, trainnn, trainpx, trainY, skeleton):
    numknots = skeleton["nknots"]
    ntrain = len(trainY)
    ntest = len(newpx)
  
    #modified data matrix for training data based projected values
    trainZ = np.array([[0.0  for col in range(3*numknots)] for row in range(ntrain) ])
    for i in range(ntrain):
        if trainpx[i] is None: #where the data point is projected to nearest knot
            trainZ[i,trainnn[i,0]] = 1
        else:
            k0 = np.min(trainnn[i,])
            k1 = np.max(trainnn[i,])
            trainZ[i, k0] = 1 - 10*trainpx[i]**3 + 15*trainpx[i]**4 -6*trainpx[i]**5
            trainZ[i, k1] = 10*trainpx[i]**3 -15*trainpx[i]**4 + 6 *trainpx[i]**5
            trainZ[i, (k0+numknots)] = trainpx[i] - 6*trainpx[i]**3 + 8*trainpx[i]**4 - 3*trainpx[i]**5
            trainZ[i, (k1+numknots)] = -4*trainpx[i]**3 + 7*trainpx[i]**4 - 3*trainpx[i]**5
            trainZ[i,(k0+ 2*numknots)] = 0.5*trainpx[i]**2 - 1.5*trainpx[i]**3 + 1.5*trainpx[i]**4 - 0.5*trainpx[i]**5
            trainZ[i,(k1+ 2*numknots)] = 0.5*trainpx[i]**3 - trainpx[i]**4 + 0.5*trainpx[i]**5
    
    #modified data matrix for test data based projected values
    testZ = np.array([[0.0  for col in range(3*numknots)] for row in range(ntest)] )
    for i in range(ntest):
        if newpx[i] is None: #where the data point is projected to nearest knot
            testZ[i,newnn[i,0]] = 1
        else:
            k0 = np.min(newnn[i,])
            k1 = np.max(newnn[i,])
            #Y0 + px(Y1-Y0), hence px(Y1), (1-px)Y0
            testZ[i, k0] = 1 - 10*newpx[i]**3 + 15*newpx[i]**4 -6*newpx[i]**5
            testZ[i, k1] = 10*newpx[i]**3 -15*newpx[i]**4 + 6 *newpx[i]**5
            testZ[i, (k0+numknots)] = newpx[i] - 6*newpx[i]**3 + 8*newpx[i]**4 - 3*newpx[i]**5
            testZ[i, (k1+numknots)] = -4*newpx[i]**3 + 7*newpx[i]**4 - 3*newpx[i]**5
            testZ[i,(k0+ 2*numknots)] = 0.5*newpx[i]**2 - 1.5*newpx[i]**3 + 1.5*newpx[i]**4 - 0.5*newpx[i]**5
            testZ[i,(k1+ 2*numknots)] = 0.5*newpx[i]**3 - newpx[i]**4 + 0.5*newpx[i]**5
  
    lmod = LinearRegression(fit_intercept= False).fit(trainZ, trainY)
    lmpred = lmod.predict(testZ)
  
    return({"trainZ": trainZ, "testZ":testZ, "model":lmod, "pred":lmpred})