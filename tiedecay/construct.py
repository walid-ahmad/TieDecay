import math
import timeit

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from tiedecay.dataset import Dataset


class TieDecayNetwork(object):
    """
    Object representing the tie strengths between nodes
    in the network.

    The user can use this class to find the tie strength
    over any given window of time contained in the dataset.

    Args:
        dataset (Dataset): dataset object
        alpha (float): tie-decay parameter
    """

    def __init__(self, dataset: Dataset, alpha: float):
        assert type(dataset) is Dataset, "Invalid type for dataset."
        self.dataset = dataset
        self.alpha = alpha
        self.history_loaded = False
        # threshold below which to filter tie strength to 0
        self.threshold = 1e-7
        return

    def compute_from_dataset(self, t: str, t_start: str = None) -> nx.DiGraph:
        """
        Compute the tie decay values over a given time window,
        using the dataset

        Args:
            t (str): time as a string that can be converted to pd.Datetime
            t_start (str): start time as a string that can be converted to pd.Datetime
                            - if not provided, the initial time in the dataset will be used
        Returns:
            B (nx.DiGraph): graph with tie strengths as edge weights

        """
        if t_start is not None:
            t_start = pd.to_datetime(t_start)
            assert t_start >= pd.to_datetime(
                self.dataset.t_first
            ), f"t_start: {t_start} must be after {pd.to_datetime(self.dataset.t_first)}"
            assert t_start < pd.to_datetime(
                self.dataset.t_last
            ), f"t_start: {t_start} must be before {pd.to_datetime(self.dataset.t_last)}"
        else:
            t_start = pd.to_datetime(self.dataset.t_first)
        t = pd.to_datetime(t)

        assert t >= t_start, f"Time t: {t} is before t_start: {t_start}"

        df = pd.DataFrame(self.dataset.adj_list)
        df.columns = ["source", "target", "time"]
        df.time = pd.to_datetime(df.time)
        df = df[df.time <= t]
        B = self._get_decay_graph(df, t)

        self.history_loaded = True
        self.B = B
        return B

    def _get_decay_graph(self, df: pd.DataFrame, t: pd.Timestamp) -> nx.DiGraph:
        """
        Get the TieDecay matrix B(t) using a dataframe

        Args:
            df (pd.DataFrame): dataframe with 'source', 'target', 'time'
            t (pd.Timestamp): timestamp at which to evaluate the td values

        Returns:
            B (nx.DiGraph): graph with tie strengths as edge weights
        """
        # get tie strength for each interaction
        df["weight"] = df.apply(
            lambda x: math.exp(-self.alpha * (t - x.time).total_seconds()), axis=1
        )

        # zero out small values
        df.weight = df.weight.mask(df.weight < self.threshold, 0)

        # sum across each pair of nodes
        td_df = df.groupby(["source", "target"]).sum().reset_index()
        td_df["weight"] = td_df["weight"]

        # construct graph
        B = nx.from_pandas_edgelist(
            td_df,
            source="source",
            target="target",
            edge_attr="weight",
            create_using=nx.DiGraph(),
        )

        return B

    def compute_centrality_trajectories_from_dataset(
        self, number_of_samples: int, centrality_method: str
    ) -> pd.DataFrame:
        """
        Sample tie-decay PageRank values from the dataset at a given resolution
        (number of samples).

        Args:
            number_of_samples (int): number of time points at which to evaluate
                                     tie strengths
            centrality_method (str): supported options:
                                        - pagerank
        Returns:
            centrality_df (pd.DataFrame): dataframe with node indices as the df index,
                                          and centrality values at each sampled
                                          time point
            sampled_times (pandas.core.indexes.datetimes.DatetimeIndex): the timestamps that were sampled
        """
        total_seconds = (
            pd.to_datetime(self.dataset.t_last) - pd.to_datetime(self.dataset.t_first)
        ).total_seconds()
        seconds_per_sample = int(total_seconds / number_of_samples)
        sampling_range = pd.date_range(
            start=self.dataset.t_first,
            end=self.dataset.t_last,
            freq=str(seconds_per_sample) + "s",
        )

        df = pd.DataFrame(self.dataset.adj_list)
        df.columns = ["source", "target", "time"]
        df.time = pd.to_datetime(df.time)
        df_init = df[df.time == pd.to_datetime(self.dataset.t_first)]
        df_init["weight"] = 1.0

        B_t = nx.from_pandas_edgelist(
            df_init,
            source="source",
            target="target",
            edge_attr="weight",
            create_using=nx.DiGraph(),
        )
        edge_list_df = nx.to_pandas_edgelist(B_t)
        centrality_values = []

        for i, t in tqdm(enumerate(sampling_range), total=len(sampling_range)):
            start_time = timeit.default_timer()
            if i == 0:
                continue
            t_prev = sampling_range[i - 1]
            # prev_edge_list_df = nx.to_pandas_edgelist(B_t)
            new_interactions_df = df[(df.time <= t) & (df.time > t_prev)]

            # decay the previous values by the same amount
            decay_factor = math.exp(-self.alpha * (t - t_prev).total_seconds())
            edge_list_df["weight"] = edge_list_df.apply(
                lambda x: x.weight * decay_factor, axis=1
            )
            #  mask small values
            edge_list_df.weight = edge_list_df.weight.mask(
                edge_list_df.weight < self.threshold, 0
            )
            # drop rows with small values from the dataframe
            edge_list_df = edge_list_df.loc[(edge_list_df.weight != 0)]

            if not new_interactions_df.empty:
                # get the tie strength values (decayed) for interactions within
                # the current window (new_interactions_df)
                B_window = self._get_decay_graph(new_interactions_df, t)

                # concat the edge lists, and sum using groupby
                new_edge_list_df = nx.to_pandas_edgelist(B_window)
                combined_edge_list_df = pd.concat(
                    [edge_list_df, new_edge_list_df], axis=0
                )
                combined_edge_list_df = (
                    combined_edge_list_df.groupby(["source", "target"])
                    .sum()
                    .reset_index()
                )
                B_t = nx.from_pandas_edgelist(
                    combined_edge_list_df,
                    source="source",
                    target="target",
                    edge_attr="weight",
                    create_using=nx.DiGraph(),
                )
                B_t.add_nodes_from(self.dataset.nodes)
            else:
                B_t = nx.from_pandas_edgelist(
                    edge_list_df,
                    source="source",
                    target="target",
                    edge_attr="weight",
                    create_using=nx.DiGraph(),
                )
                B_t.add_nodes_from(self.dataset.nodes)
            centrality_vals_t = nx.pagerank(B_t)
            centrality_values.append(centrality_vals_t)
            edge_list_df = nx.to_pandas_edgelist(B_t)
        centrality_df = pd.DataFrame(centrality_values).transpose()
        centrality_df.columns = sampling_range[1:]
        return centrality_df
