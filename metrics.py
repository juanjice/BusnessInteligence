import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getDistances(D):
    if type(D) == pd.DataFrame:
        D = D.values
    n = len(D)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            W[i,j] = np.linalg.norm(D[i] - D[j])
    return W

def plotClusters(D, C, dimX, dimY, dimZ = None, ax = None):
    isDF = type(D) == pd.DataFrame
    
    # Extract labels for the dimensions if this is a dataframe
    labelX = D.columns[dimX] if isDF else dimX
    labelY = D.columns[dimY] if isDF else dimY
    labelZ = D.columns[dimZ] if isDF and not dimZ is None else dimZ
    
    # transform data into an nd-array
    if type(D) == pd.DataFrame:
        D = D.values
    
    # compute the cluster identifiers
    clusterIDs = np.unique(C)
    
    # determine the axis object for the plots
    plot3d = not dimZ is None
    if plot3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        if ax is None:
            fig, ax = plt.subplots()
    
    # scatter the points of each cluster into the axis objects
    for ci in clusterIDs:
        indices = np.where(C == ci)[0]
        if plot3d:
            ax.scatter(D[indices,dimX], D[indices,dimY], D[indices,dimZ])
        else:
            ax.scatter(D[indices,dimX], D[indices,dimY])
    
    # Attach axis labels
    ax.set_xlabel(labelX)
    ax.set_ylabel(labelY)
    if plot3d:
        ax.set_zlabel(labelZ)
        
def getW(W, U, V):
    s = 0
    for u in U:
        for v in V:
            s += W[max(u,v), min(u,v)]
    return s

def getWIn(D, C):
    clusters = np.unique(C)
    W = getDistances(D)
    s = 0
    for c in clusters:
        cIndices = np.where(C == c)[0]
        s += getW(W, cIndices, cIndices)
    return s / 2

def getWOut(D, C):
    clusters = np.unique(C)
    W = getDistances(D)
    s = 0
    for c in clusters:
        cIndices1 = np.where(C == c)[0]
        cIndices2 = np.where(C != c)[0]
        s += getW(W, cIndices1, cIndices2)
    return s / 2

def getNIn(C):
    clusters = np.unique(C)
    s = 0
    for c in clusters:
        ni = np.count_nonzero(C == c)
        s += ni * (ni - 1)
    return int(s / 2)

def getNOut(C):
    clusters = np.unique(C)
    s = 0
    for c1 in clusters:
        n1 = np.count_nonzero(C == c1)
        for c2 in clusters:
            if c1 != c2:
                n2 = np.count_nonzero(C == c2)
                s += n1 * n2
    return int(s / 2)

def getBetaCV(D, C):
    return getWIn(D, C) / getWOut(D, C) * getNOut(C) / getNIn(C)

def getCIndex(D, C):
    NIn = getNIn(C)
    distAsVector = np.sort(np.ravel(getDistances(D)))
    
    WIn = getWIn(D, C)
    WMin = np.sum(distAsVector[:NIn])
    WMax = np.sum(distAsVector[-1 * NIn:])
    
    return (WIn - WMin) / (WMax - WMin)

def getNormalizedCut(D, C):
    clusters = np.unique(C)
    W = getDistances(D)
    s = 0
    allIndices = range(len(C))
    for c in clusters:
        cIndices1 = np.where(C == c)[0]
        cIndices2 = np.where(C != c)[0]
        nom = getW(W, cIndices1, cIndices2)
        den = getW(W, cIndices1, allIndices)
        s += nom / den
    return s
    
def getModularity(D, C):
    clusters = np.unique(C)
    W = getDistances(D)
    s = 0
    allIndices = range(len(C))
    WVV = 2 * (getWIn(D, C) + getWOut(D, C))
    for c in clusters:
        cIndices = np.where(C == c)[0]
        s += (getW(W, cIndices, cIndices) / WVV) - (getW(W, cIndices, allIndices) / WVV) ** 2
    return s
    
def getDunn(D, C):
    if type(D) == pd.DataFrame:
        D = D.values
    
    clusters = np.unique(C)
    
    # compute the minimum
    WMinOut = np.inf
    for ci in clusters:
        cIndices1 = np.where(C == ci)[0]
        for cj in clusters:
            if ci < cj:
                cIndices2 = np.where(C == cj)[0]
                for xa in cIndices1:
                    for xb in cIndices2:
                        WMinOut = min(WMinOut, np.linalg.norm(D[xa] - D[xb]))
    
    # compute the maximum
    WMaxIn = -1
    for ci in clusters:
        cIndices = np.where(C == ci)[0]
        for xa in cIndices:
            for xb in cIndices:
                WMaxIn = max(WMaxIn, np.linalg.norm(D[xa] - D[xb]))
    
    return WMinOut / WMaxIn

def getDaviesBouldin(D, C):
    if type(D) == pd.DataFrame:
        D = D.values
    DBSum = 0
    clusters = np.unique(C)
    for ci in clusters:
        cIndices1 = np.where(C == ci)[0]
        mean1 = np.mean(D[cIndices1])
        std1 = np.std(D[cIndices1])
        pairScore = -1
        for cj in clusters:
            if ci < cj:
                cIndices2 = np.where(C == cj)[0]
                mean2 = np.mean(D[cIndices2])
                std2 = np.std(D[cIndices2])
                pairScore = max(pairScore, (std1 + std2) / np.linalg.norm(mean1 - mean2))
        DBSum += pairScore
    return DBSum

def getSilhouette(D, C):
    
    if type(D) == pd.DataFrame:
        D = D.values
        
    W = getDistances(D)
    clusters = np.unique(C)
    
    # run over all points
    sCoeff = 0
    for i, x in enumerate(D):
        
        c = C[i]
        indicesOfOtherPointsInSameCluster = np.where(C == c)[0]
    
        # compute intra-cluster distance
        muIn = 0
        for j in indicesOfOtherPointsInSameCluster:
            muIn += W[max(i,j),min(i,j)]
        muIn /= (len(indicesOfOtherPointsInSameCluster) - 1)

        # compute minimum inter-cluster distance
        muOutMin = np.inf
        for cj in clusters:
            if c != cj:
                s = 0
                pointsInCluster = np.where(C == cj)[0]
                for y in pointsInCluster:
                    s += W[max(i,y), min(i,y)]
                s /= len(pointsInCluster)
                muOutMin = min(muOutMin, s)
        
        # add score of this point to overall score
        sCoeff += (muOutMin - muIn) / max(muOutMin, muIn)
    
    # return score normalized by the number of points
    return sCoeff / len(D)

def getMetric(D, C, metric):
    if metric == "beta":
        return getBetaCV(D, C)
    elif metric == "cindex":
        return getCIndex(D, C)
    elif metric == "nc":
        return getNormalizedCut(D, C)
    elif metric == "mod":
        return getModularity(D, C)
    elif metric == "dunn":
        return getDunn(D, C)
    elif metric == "db":
        return getDaviesBouldin(D, C)
    elif metric == "sil":
        return getSilhouette(D, C)
    else:
        raise Exception("Unknown metric " + str(metric))