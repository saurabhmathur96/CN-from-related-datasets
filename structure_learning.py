from typing import *
import numpy as np
from scipy.special import gammaln
from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianNetwork
from cutset_network import CutsetNetworkNode, LeafNode, OrNode
from estimators import ProbabilityEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from dataset import Dataset
# from pgmpy.estimators import BicScore



def learn_structure(D: Dataset, learn_leaf, score, min_variables: int = 5,
                    min_score: float = 0.1, min_instances=10) -> CutsetNetworkNode:
   #  print (len(D))
    if len(D.scope) < min_variables or len(D) < min_instances:
        leaf = learn_leaf(D)
        return leaf
    else:
        candidates = D.scope
        scores = score(D, candidates)
        i = np.argmax(scores)
        # print (scores[i])
        best_cut = candidates[i]
        if scores[i] < min_score:
            leaf = learn_leaf(D)
            return leaf
        else:
            Ds = D.split(best_cut)
            kwargs = dict(learn_leaf=learn_leaf, score=score,
                          min_variables=min_variables,
                          min_score=min_score, min_instances=min_instances)
            children = [learn_structure(Di, **kwargs)
                        for Di in Ds]
            est = ProbabilityEstimator(D)
            p = est.probability([best_cut])
            weights = np.array([p[value]
                               for value in np.ndindex(D.r[best_cut])])
            return OrNode(D.scope, D.r, best_cut, weights, children)


def learn_cnet_stump(D: Dataset, v: str, learn_leaf) -> OrNode:
    Ds = D.split(v)
    children = [learn_leaf(Di)
                for Di in Ds]
    est = ProbabilityEstimator(D)
    p = est.probability([v])
    weights = np.array([p[value] for value in np.ndindex(D.r[v])])
    return OrNode(D.scope, D.r, v, weights, children)


def fully_factorized_leaf(D: Dataset, estimate_parameters=False):
    bn = BayesianNetwork()
    bn.add_nodes_from(D.scope)
    if not estimate_parameters:
        cpds = [
            TabularCPD(
                variable=v, variable_card=D.r[v],
                values=[[1 / D.r[v]] for _ in range(D.r[v])])
            for v in D.scope]
        bn.add_cpds(*cpds)
    else:
        df = D.as_df()
        state_names = {v: list(range(D.r[v])) for v in D.scope}
        bn.fit(df, BayesianEstimator, prior_type="dirichlet",
               pseudo_counts=1, state_names=state_names)

    return LeafNode(D.scope, D.r, bn)


def learn_chow_liu_leaf(D: Dataset) -> LeafNode:
    df = D.as_df()
    state_names = {v: list(range(D.r[v])) for v in D.scope}
    if len(df) == 0:
        return fully_factorized_leaf(D)

    dag = TreeSearch(df, state_names=state_names).estimate(
        estimator_type="chow-liu", show_progress=False)
    bn = BayesianNetwork()
    bn.add_nodes_from(D.scope)
    bn.add_edges_from(dag.edges())

    bn.fit(df, BayesianEstimator, prior_type="dirichlet",
           pseudo_counts=1, state_names=state_names)
    return LeafNode(D.scope, D.r, bn)
