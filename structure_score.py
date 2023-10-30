from dataset import Dataset, GroupedDataset
from typing import *
import numpy as np
from sklearn.metrics import mutual_info_score
from structure_learning import learn_cnet_stump, learn_chow_liu_leaf
from cutset_network import depth_first_order
from parameter_learning import estimate_parameters, set_parameters


def mi_score(D: Dataset, candidates: List[str]):
    df = D.as_df()
    if len(df) == 0:
        return np.zeros(len(candidates)) - np.inf

    score = np.zeros((len(candidates),))
    for i, u in enumerate(candidates):
        score[i] += np.sum([mutual_info_score(df[u], df[v])
                            for v in D.scope if v != u])

    return score


def bic_score(D: Dataset, candidates: List[str]):
    score = np.zeros((len(candidates),))
    for i, u in enumerate(candidates):
        or_node = learn_cnet_stump(D, u, learn_chow_liu_leaf)
        ll = or_node(D, log=True)
        n_parameters = sum(n.n_parameters for n in depth_first_order(or_node))
        penalty = np.log(len(D))*n_parameters/2
        score[i] = ll - penalty
    leaf = learn_chow_liu_leaf(D)
    penalty = np.log(len(D))*leaf.n_parameters/2
    base = leaf(D, log=True) - penalty
    return score - base


def bic_score2(D: Dataset, candidates: List[str]):
    denom = np.sum([np.log(D.r[v]) for v in D.scope])

    score = np.zeros((len(candidates),))
    for i, u in enumerate(candidates):
        or_node = learn_cnet_stump(D, u, learn_chow_liu_leaf)

        ll = or_node(D, log=True)
        n_parameters = sum(n.n_parameters for n in depth_first_order(or_node))
        penalty = (np.log(len(D))*n_parameters/2)/denom

        score[i] = ll - penalty
    leaf = learn_chow_liu_leaf(D)
    penalty = (np.log(len(D))*leaf.n_parameters/2)/denom
    base = leaf(D, log=True) - penalty
    return score - base


def group_bic_score(D: GroupedDataset, candidates: List[str]):
    score = np.zeros((len(candidates),))
    for i, u in enumerate(candidates):
        or_node = learn_cnet_stump(D, u, learn_chow_liu_leaf)
        ll = 0
        for dataset in D.datasets:
            params = estimate_parameters(or_node, dataset, leaf_only=True)
            set_parameters(or_node, params, leaf_only=True)
            ll += or_node(dataset, log=True)

        child_parameters = sum(n.n_parameters for n in or_node.children)
        penalty = np.log(len(D)) * or_node.n_parameters / 2 + np.log(
            len(D.datasets)) * len(D.datasets) * child_parameters / 2

        score[i] = ll - penalty
    leaf = learn_chow_liu_leaf(D)
    ll = 0
    for dataset in D.datasets:
        if len(dataset) == 0:
            continue
        param = estimate_parameters(leaf, dataset)
        set_parameters(leaf, param)
        ll += leaf(dataset, log=True)
    penalty = np.log(len(D.datasets))*len(D.datasets)*leaf.n_parameters/2
    base = ll - penalty
    return score - base


def group_bic_score2(D: GroupedDataset, candidates: List[str]):
    denom = np.sum([np.log(D.r[v]) for v in D.scope])  # + len(D.datasets)
    score = np.zeros((len(candidates),))
    for i, u in enumerate(candidates):
        or_node = learn_cnet_stump(D, u, learn_chow_liu_leaf)
        ll = 0
        for dataset in D.datasets:
            params = estimate_parameters(or_node, dataset, leaf_only=True)
            set_parameters(or_node, params, leaf_only=True)
            ll += or_node(dataset, log=True)

        child_parameters = sum(n.n_parameters for n in or_node.children)
        penalty = (
            np.log(len(D)) * or_node.n_parameters / 2 + np.log(len(D.datasets)) *
            len(D.datasets) * child_parameters / 2) / denom

        score[i] = ll - penalty
    leaf = learn_chow_liu_leaf(D)
    ll = 0
    for dataset in D.datasets:
        if len(dataset) == 0:
            continue
        param = estimate_parameters(leaf, dataset)
        set_parameters(leaf, param)
        ll += leaf(dataset, log=True)
    penalty = (np.log(len(D.datasets))*len(D.datasets)
               * leaf.n_parameters/2)/denom
    base = ll - penalty
    return score - base
