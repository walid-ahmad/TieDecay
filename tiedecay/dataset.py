import operator
import os
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm


class Dataset(object):
    """
    Object for a data source that exists in the form of a list:
        [
            (source, target, timestamp),
            (source, target, timestamp),
            ...
        ]

    Initialize using this `data`, along with an optional `node_mapping`.
    The `node_mapping` will be used to calculate the total number of nodes.
    Otherwise, the number of active nodes (nodes found in `data` as sources
    or targets) will be used. Note that this can affect the computation
    of TieDecay PageRank values downstream.

    """

    def __init__(self, data: List, node_mapping: Dict[int, Any] = None):
        """
        Args:
            data (list): list of timestamped edges (source, target, timestamp)
            node_mapping (dict): optional mapping of node ids to node names, or other metadata
        """
        self.adj_list = data
        self.node_mapping = node_mapping
        self._load_data()
        return

    @property
    def nodes(self):
        if self.node_mapping is not None:
            self._nodes = set(self.node_mapping.keys())
        else:
            self._nodes = self.active_nodes
        return self._nodes

    @property
    def num_nodes(self):
        if self.node_mapping is not None:
            self._num_nodes = len(set(self.node_mapping.keys()))
        else:
            self._num_nodes = len(self.active_nodes)
        return self._num_nodes

    def _load_data(self):
        """
        Load the data into computationally-friendly format
        """
        self.num_interactions = len(self.adj_list)
        self.sort_interactions()
        self.sources = [x[0] for x in self.adj_list]
        self.targets = [x[1] for x in self.adj_list]
        self.timestamps = [x[2] for x in self.adj_list]
        self.active_nodes = set(self.sources + self.targets)
        self.num_active_nodes = len(self.active_nodes)

        # Convert the adjacency list to a dictionary with timestamps as keys
        print("Converting to dictionary...")
        self.adj_dict = defaultdict(list)
        for i in tqdm(self.adj_list):
            t = pd.to_datetime(i[2])
            self.adj_dict[t].append((i[0], i[1]))
        print("Done.")

    def sort_interactions(self, ascending: bool = True):
        """
        Sort the data in-place based on timestamp

        Args:
            ascending: bool indicating whether to sort first-to-last
        """
        print("Sorting data...")
        self.adj_list = sorted(
            self.adj_list, key=operator.itemgetter(2), reverse=(not ascending)
        )
        self.t_first = self.adj_list[0][-1]
        self.t_last = self.adj_list[-1][-1]
        print("Data sorted!")
        return
