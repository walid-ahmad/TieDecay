#!/usr/bin/env python

"""\

Evaluate decay matrix and tie-decay centrality scores at evenly spaced
time points

Give as keyword input on the command line the half life to evaluate in hours,
e.g.:

$python main.py 24

for a 24 hour half life. You may also input multiple half lives, separated
by spaces, e.g.:

$python main.py 1 24

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


    # create storage variable for centrality scores and iterations
    TD_PRs_t = np.zeros(shape=(nb_users, nb_timepoints), dtype=np.float32)
    TD_PRs_u = np.zeros(shape=(nb_users, nb_timepoints), dtype=np.float32)
    PR_iterst = np.zeros(shape=(1, nb_timepoints), dtype=np.uint8)
    PR_itersu = np.zeros(shape=(1, nb_timepoints), dtype=np.uint8)

    # initialize B_t to sparse matrices of zeros, and pr vector to uniform
    B_t = sparse.csr_matrix((nb_users,nb_users), dtype=np.float32)
    pr_t = dict.fromkeys(range(nb_users), 1.0 / nb_users)

    # set threshold for eliminating small values
    threshold = 10**(-7)

    # set associated decay values
    alphas = np.log(2)/half_lives/3600


    # now iterate through the time range
    print "Alphas progress: "
    for hl,alpha in tqdm(zip(half_lives, alphas)):
        print "\n Sampling range progress: "
        for i,t in tqdm(enumerate(sampling_range)):
            print " \n Sample progress: "
            for delta_seconds in tqdm(range(seconds_per_sample)):
                tstamp = t+pd.Timedelta(str(delta_seconds)+'s')
                A_t = tieDecayMat.getdirAdjNow(dataAdjDict, t=tstamp, n=nb_users)
                B_t = tieDecayMat.decay(A_t, B_t, alpha, threshold)

            # create network with B_t as adj matrix
            G = nx.from_scipy_sparse_matrix(B_t, create_using=nx.DiGraph())

            # calculate PageRank and # of iterations required
            pr_t, n_itert = prcust.pagerank(G, nstart=pr_t)
            pr_u, n_iteru = prcust.pagerank(G)

            # store these results
            for u, score in pr_t.items():
                TD_PRs_t[u][i] = float(score)
            for u, score in pr_u.items():
                TD_PRs_u[u][i] = float(score)
            PR_iterst[i] = n_itert
            PR_itersu[i] = n_iteru

            vars_to_save = [TD_PRs_t, TD_PRs_u, PR_iterst, PR_itersu]
            names_to_save = ['TD_PRs_t', 'TD_PRs_u', 'PR_iterst', 'PR_itersu']
            for name, v in zip(names_to_save, vars_to_save):
                storage.save_obj(v, name+'_alpha_'+str(hl))
