"""
Set of functions responsible for processing an adjacency list, in the form of a
[(source, target, time), ... ] list and creating the Tie-Decay matrix

"""

import numpy as np
import pandas as pd
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

    And obtain the directed adjacency matrix for time t.

    Specify the number of nodes as input.

    """
    t = pd.to_datetime(t)
    A_t = sparse.csr_matrix((n, n), dtype=np.int8)
    if t in adjDict.keys():
        for i in adjDict[t]:
            row = np.array([i[0]])
            cols = np.array([i[1]])
            data = np.array([1])
            A = sparse.csr_matrix((data, (row, col)), shape = (n, n)
                                    dtype=np.int8)
            A_t += A
    else:
        pass

    return A_t
