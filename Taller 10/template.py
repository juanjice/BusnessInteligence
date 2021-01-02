import numpy as np
import pandas as pd
import itertools

# reads in a transaction data file and returns a list of lists.
# Each element in the output list is a transaction. This is a list of items contained in this transaction
def readTransactionalDatabase(file):
    db = []
    with open(file) as f:
        lineList = f.readlines()
        for line in lineList:
            li = line.split(" ")
            row = []
            for v in li:
                if v != "\n":
                    row.append(int(v))
            db.append(row)
    return db


# Computes the difference list between list li1 and li2
def diff(li1, li2):
    return [x for x in li1 if not x in li2]


# Computes the set of all subsets (lists) of li
def powerset(li):
    ps = itertools.chain.from_iterable(itertools.combinations(li, r) for r in range(0, len(li) + 1))
    pl = []
    for e in reversed(list(ps)):
        pl.append(list(e))
    return pl


def eclat(db, minsup):
    
    # compute single item itemsets and the corresponding transactions
    pset = {}
    for tIndex, t in enumerate(db):
        for i in t:
            if not i in pset:
                pset[i] = []
            pset[i].append(tIndex)
    
    # initialize the set P of children and the set of frequent sets
    
    F = []
    P = [([i],set(pset[i])) for i in sorted(pset.keys()) if len(pset[i]) >= minsup]    
    eclatRec(P, minsup, F)
    return F

def eclatRec(P, minsup, F): 
    for indexA, (Xa, tXa) in enumerate(P):
        F.append((Xa, len(tXa))) # mark this itemset as frequent
        
        # compute children for this node
        Pa = []
        for Xb, tXb in P[indexA+1:]:
            tXab = tXa & tXb
            if len(tXab) >= minsup:
                Pa.append((sorted(list(set(Xa + Xb))), tXab))
        
        # if there are children, recurse
        if Pa:
            eclatRec(Pa,minsup,F)

def createAssociationRules(F, minconf):
    rules = []
    
    # run over all frequent itemsets of size more than one
    for Z, supZ in [fEntry for fEntry in F if len(fEntry[0]) > 1]:
        
        # compute all possible rule premises for this set
        A = sorted(powerset(Z), key=lambda l : len(l), reverse=True)
        A.remove([])
        A.remove(Z)
        
        # run over all possible premises that have not been discarded yet
        while A:
            X = A.pop(0)
            supX = [xEntry for xEntry in F if xEntry[0] == X][0][1]
            c = supZ / supX
            if c >= minconf:
                rules.append(((X, diff(Z, X)), supZ, c))
            else:
                prunables = powerset(X)
                for prunable in prunables:
                    if prunable in A:
                        A.remove(prunable)
    return rules

def getStrongRules(db, minsup, minconf):
    return createAssociationRules(eclat(db, minsup), minconf)
