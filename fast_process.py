#!/usr/bin/env python

"""\

Fast-process data set by applying decay and summation directly at discrete
time points

Record only PageRank scores (not iterations)

"""
import sys
import operator
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from tqdm import *

import tieDecayMat
import storage
import prcust

if __name__ == "__main__":
    try:
        half_lives = sys.argv[1:]
        half_lives = [int(h) for h in half_lives]
        half_lives = np.array(half_lives)
        print "Half lives: ", half_lives
    except IndexError:
        print "Please provide the half life (or half lives) as argument(s)"
        sys.exit()


    dataPath = '/Users/walid/Dropbox/Tie_Decay_Centrality/Data/NHSadjList'
    usersPath = '/Users/walid/Dropbox/Tie_Decay_Centrality/Data/NHSusersDict'

    print "Loading data..."
    dataAdjList = storage.load_obj(dataPath)
    usersDict = storage.load_obj(usersPath)
    print "Data loaded!"
    nb_users = len(usersDict)

    # sort list by time to get start and end times
    print "Sorting data..."
    dataAdjList = sorted(dataAdjList, key=operator.itemgetter(2))
    print "Data sorted!"
    t_initial = dataAdjList[0][2]
    t_final = dataAdjList[-1][2]

    # convert to dictionary for O(1) lookup
    print "Converting to dictionary..."
    dataAdjDict = tieDecayMat.convert_List_to_Dict(dataAdjList)
    print "Converting to dictionary done!"

    # specify the number of timepoints we'll sample the data at
    nb_timepoints = 1000

    total_seconds = (pd.to_datetime(t_final) - pd.to_datetime(t_initial)).total_seconds()
    seconds_per_sample = int(total_seconds) / nb_timepoints

    sampling_range = pd.date_range(start=t_initial, end=t_final,
                                    freq=str(seconds_per_sample)+'s')
    sampling_range_plus = sampling_range[1:]

    # set threshold for eliminating small values
    threshold = 10**(-7)

    # set associated decay values
    alphas = np.log(2)/half_lives/3600

    # create storage variable for centrality scores and iterations
    TD_PRs = np.zeros(shape=(nb_users, nb_timepoints), dtype=np.float32)
    B = sparse.csr_matrix((nb_users,nb_users), dtype=np.float32)

    # now iterate through the time range
    print "Alphas progress: "
    for hl,alpha in tqdm(zip(half_lives, alphas)):
        print "\n Sampling range progress: "
        for i,t in tqdm(enumerate(sampling_range_plus)):
            t1 = t-pd.Timedelta(str(seconds_per_sample)+'s')
            t2 = t

            B = tieDecayMat.getDecayAdjBatch(dataAdjDict, t1, t2, B, nb_users, alpha)
            B = B.multiply(B>=threshold)

            # create network with B_t as adj matrix
            G = nx.from_scipy_sparse_matrix(B, create_using=nx.DiGraph())
            pr_t = nx.pagerank(G)

            for u, score in pr_t.items():
                TD_PRs[u][i] = float(score)

            storage.save_obj(TD_PRs, 'TD_PRs'+'_alpha_'+str(hl))
