"""/

Evaluate decay matrix and tie-decay centrality scores at evenly spaced
time points

"""
import operator
import pandas as pd
import networkx as nx
from tqdm import *

import tieDecayMat
import storage
import prcust



if __name__ == "__main__":
    dataPath = '/Users/walid/Dropbox/Tie_Decay_Centrality/Data/NHSadjList'
    usersPath = '/Users/walid/Dropbox/Tie_Decay_Centrality/Data/NHSusersDict'
    dataAdjList = storage.load_obj(dataPath)
    usersDict = storage.load_obj(usersPath)
    nb_users = len(usersDict)

    # sort list by time to get start and end times
    dataAdjList = sorted(dataAdjList, key=operator.itemgetter(2))
    t_initial = dataAdjList[0][2]
    t_final = dataAdjList[-1][2]

    # convert to dictionary for O(1) lookup
    dataAdjDict = tieDecayMat.convert_List_to_Dict(dataAdjList)

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
    B_t = sparse.csr_matrix((nb_users,nb_users), dtype=np.int8)
    pr_t = dict.fromkeys(range(nb_users), 1.0 / nb_users)

    # set threshold for eliminating small values
    threshold = 10**(-7)

    # set half lives and associated decay values
    half_lives = 3600*np.array([1, 12, 24, 72, 168])
    alphas = np.log(2)/half_lives


    # now iterate through the time range
    for i,t in tqdm(enumerate(sampling_range)):
        for delta_seconds in range(seconds_per_sample):
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
            storage.save_obj(v, name)
