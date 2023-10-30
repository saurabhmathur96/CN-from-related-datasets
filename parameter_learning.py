from typing import List
import numpy as np
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from dataset import Dataset
from cutset_network import CutsetNetworkNode, LeafNode, depth_first_order
from estimators import ProbabilityEstimator

def estimate_parameters(cnet: CutsetNetworkNode, D: Dataset, s: float = 1, leaf_only=False):
    parameters = []
    
    if leaf_only:
        iterator = ((node, Di) for node, Di in depth_first_order(cnet, D) if isinstance(node, LeafNode))
    else:
        iterator = depth_first_order(cnet, D) 
    for node, Di in iterator:
        if isinstance(node, LeafNode):
            if len(Di) == 0:
                cpds = []
                for cpd in node.bn.get_cpds():
                    new_cpd = cpd.copy()
                    new_cpd.values = np.ones_like(new_cpd.values)
                    new_cpd.normalize(inplace=True)
                    cpds.append(new_cpd)
                parameters.append(cpds)
            else:
                state_names = { v: list(range(Di.r[v])) for v in Di.scope}
                est = BayesianEstimator(node.bn, Di.as_df(), state_names=state_names)
                cpds = est.get_parameters(prior_type="dirichlet", pseudo_counts=s)
                parameters.append(cpds)
        else:
            est = ProbabilityEstimator(Di, s=s)
            prob = est.probability([node.v])
            weights = [prob[value] for value in np.ndindex(node.r[node.v])]
            parameters.append(weights)
    return parameters

def set_parameters(cnet: CutsetNetworkNode, parameters: List, leaf_only=False):
    # sets parameters inplace

    if leaf_only:
        leaves = (node for node in depth_first_order(cnet) if isinstance(node, LeafNode))
        iterator = ((node, param) for node, param in zip(leaves, parameters))
    else:
        iterator = zip(depth_first_order(cnet), parameters) 
    for node, param in iterator:
        if isinstance(node, LeafNode):
            node.bn.remove_cpds(*node.bn.get_cpds())
            node.bn.add_cpds(*param)
        else:
            node.weights = param
    return cnet