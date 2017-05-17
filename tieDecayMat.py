"""
Set of functions responsible for processing an adjacency list, in the form of a
[(source, target, time), ... ] list and creating the Tie-Decay matrix

"""

import numpy as np
import pandas as pd
from scipy import sparse
import math
from tqdm import *

def convert_List_to_Dict(adjList):
    """\
    Convert adjacency list in the form:

    [(source, target, time), (source, target time), ...]

    to an adjacency dictionary, with timestamps as keys:

    { t: (source, target), ... }

    This will improve speed when looking for interactions at a
    particular timestamp.

    """
    adjDict = {pd.to_datetime(i[2]): [] for i in adjList}
    for i in tqdm(adjList):
        t = pd.to_datetime(i[2])
        adjDict[t].append((i[0], i[1]))
    return adjDict


def getdirAdjNow(adjDict, t, n):
    """\
    Input an unweighted adjacency dictionary in the form:

    { t: (source, target), ... }

    and obtain the directed adjacency matrix for time t.

    Specify the number of nodes as input.

    """
    t = pd.to_datetime(t)
    A_t = sparse.csr_matrix((n, n), dtype=np.int8)
    if t in adjDict.keys():
        for i in adjDict[t]:
            row = np.array([i[0]])
            col = np.array([i[1]])
            data = np.array([1])
            A = sparse.csr_matrix((data, (row, col)), shape = (n,n),
                                    dtype=np.int8)
            A_t += A
    else:
        pass

    return A_t

def getDfdirAdjNow(df, t, n):
    """\
    Input an unweighted adjacency pandas DataFrame() that has the columns:
    ['Time', 'Source', 'Target']

    and obtain the directed adjacency matrix for time t.

    Specify the number of nodes as input.
    """
    t = pd.to_datetime(t)
    A_t = sparse.csr_matrix((n,n), dtype=np.int8)

    try:
        sources = list(df.loc[df['Time'] == str(t)]['Source'])
        targets = list(df.loc[df['Time'] == str(t)]['Target'])
        data = np.array([1])

        for s, t in zip(sources, targets):
            row = np.array([s])
            col = np.array([t])
            A = sparse.csr_matrix((data, (row,col)), shape=(n,n),
                                    dtype=np.int8)
            A_t = A_t + A

    except:
        pass

    return A_t

def decay(A_t, B_tminus1, alpha, threshold):
    """\
    Apply decay to the tie-strength matrix.

    Input A(t) the adjacency matrix at time t, B(t-1) the
    matrix of tie strengths at the previous time step, the decay rate
    alpha, and the threshold for eliminating small values for computational
    efficiency. Any values in the tie strength matrix that are below the value
    given by threshold will be set to 0.

    Returns the matrix of tie-strengths at time t.

    """

    # Calculate and apply the decay on B_tminus1
    H_t = B_tminus1.multiply(math.exp(-alpha))

    # eliminate small values from the matrix by setting them to 0
    H_t = H_t.multiply(H_t>=threshold)

    # add on new values
    B_t = H_t + A_t

    return B_t
