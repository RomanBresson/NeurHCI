import itertools
from math import factorial
import torch
import torch.nn as nn
from .aggregators import CI2Add

class HCI(nn.Module):
    """
        This module implements a hierarchical Choquet integral, i.e. a computational directed-acyclic graph whose
        nodes are Choquet integrals.
    """
    def __init__(self, hierarchy):
        """
            The hierarchy is given as a dictionary {id_of_parent (int): [ids_of_children]}
            The hierarchy must be acyclic.
            The hierarchy can have several roots (outputs). In this case, the outputs are returned by ascending order of Ids.
            The roots are the node without a parent. The leaves are the nodes without children.
            The inputs are taken by ascending order of leaves.
            
            Examples of hierarchy:
                {-1:[0,3], 3:[1,2]} represents a tree with two aggregators. 
                        -Leaves are 0, 1 and 2. 1 and 2 are aggregated by 3, then 0 and 3 are aggregated at the root.
                        -The only root is -1. 
                        -3 is an intermediate node.
                
                {-1:[0,3], 3:[1,2], 4:[1,2,3]} represents a tree with three aggregators (-1, 3, 4). 
                        -There are two roots (-1 and 4), so the output will be two-dimensional.
                        -3 is an intermediate node.
                
                {-1:[0,3], 3:[1,2], 4:[3]} represents a tree with three aggregators (-1, 3, 4). 
                        -Leaves are 0, 1 and 2. 1 and 2 are aggregated by 3, then 0 and 3 are aggregated at the -1.
                        -4 copies the value of 3, as a CI with a single input is an identity function. This allows to use
                        intermediate values as outputs.
                        -There are two roots (-1 and 4), so the output will be two-dimensional.
                        -3 is an intermediate node.

                {-1:[4,5], 4:[0,1,2], 5:[2,3]} represents a DAG, with three aggregators.
                        -Leaves are 0 to 4.
                        -The only root is -1.
                        -4 and 5 are intermediate nodes. Note that 2 is used by both of them.
                
                {-1:[4,5], 4:[0,1,5], 5:[2,3,4]} represents a graph with a cycle (4 and 5 aggregate each other) and is
                thus not valid.
            
            Input for batch size m is a tensor of shape (m, l), with l the number of leaves.
            Output is of shape (m, r) with r the number of roots.
        """
        super(HCI, self).__init__()
        self.hierarchy = hierarchy
        all_aggregators = list(hierarchy.keys())
        all_children_list = list(itertools.chain.from_iterable(hierarchy.values()))
        all_children = set(all_children_list)
        self.is_tree = len(all_children_list)==len(all_children)
        leaves = [node for node in all_children if node not in all_aggregators]
        roots = [agg for agg in all_aggregators if agg not in all_children]
        self.leaves = sorted(leaves)
        self.roots = sorted(roots)
        self.aggregators = all_aggregators
        self.dim = len(self.leaves)
        self.set_heightmap()
        self.CIs = nn.ModuleDict({str(agg): CI2Add(dimension=len(children)) for agg,children in self.hierarchy.items()})
        self.global_winter_values = None

    def set_heightmap(self):
        """
            Computes the height (longest distance to a leaf) of each node and builds a height map.
            Must be rebuilt if the hierarchy changes.
            Is used mostly for an efficient forward propagation.
        """
        self.parents = {r:[] for r in self.roots}
        for parent,children in self.hierarchy.items():
            for child in children:
                self.parents.setdefault(child, []).append(parent)
        self.heights = {c:0 for c in self.leaves+self.aggregators}
        last_treated = set(self.leaves)
        h = 1
        while True:
            if h>len(self.hierarchy)+1:
                raise ValueError("Hierarchy contains a loop")
            to_treat = set(itertools.chain.from_iterable(self.parents[c] for c in last_treated))
            if not to_treat:
                break
            for t in to_treat:
                self.heights[t] = h
            h+=1
            last_treated = to_treat
        self.heightmap = {i:[] for i in range(max(self.heights.values())+1)}
        for n,h in self.heights.items():
            self.heightmap[h].append(n)
        self.max_height = max(self.heightmap.keys())

    def forward(self, x):
        """
            Assumes the heightmap is up to date, as the propagation is done level by level.
        """
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        self.node_values = {n:None for n in self.aggregators}
        for i,l in enumerate(self.leaves):
            self.node_values[l] = x[:,i:i+1]
        for h in range(1,self.max_height+1):
            for n in self.heightmap[h]:
                children_values = [self.node_values[k] for k in self.hierarchy[n]]
                inputs = torch.cat(children_values, axis=1)
                self.node_values[n] = self.CIs[str(n)](inputs)
        return(torch.cat([self.node_values[r] for r in self.roots], axis=1))
    
    def compute_winter_values_global(self, starting_node=None):
        """
            starting_node: which node's value is to be explained by its descendents.
            If none is given, the root (which needs to be single then) is selected.

            The subgraph yielded by the starting node's descendants must be a tree.
        """
        if starting_node is None:
            assert len(self.roots)==1, "The node for which to compute the Shapley values must be given"
            starting_node = self.roots[0]
        if starting_node in self.leaves:
            return({starting_node: torch.tensor(1.)})
        next_to_treat = [starting_node]
        shapley_values = {}
        global_winter_values = {starting_node: 1.}
        seen_nodes = [starting_node]
        while next_to_treat:
            current = next_to_treat.pop(0)
            shapley_values[current] = self.CIs[str(current)].compute_shapley_values_global()
            for child,shap in zip(self.hierarchy[current], shapley_values[current]):
                global_winter_values[child] = global_winter_values[current]*shap
                if child in seen_nodes:
                    raise ValueError(f"The subgraph yielded by the starting node's descendants must be a tree. {current} has at least two parents in this tree.")
                seen_nodes.append(child)                
                if child not in self.leaves:
                    next_to_treat.append(child)
        return(global_winter_values)

    def compute_winter_values_local(self, x, y, starting_node=None):
        #TODO
        ...

class HCI2layers(HCI):
    """
        A tree-HCI with a single root and a single intermediate layer. 
        Each node in the intermediate layer has the same number of leaves (except the last one)
        The root aggregates all intermediate nodes
    """
    def __init__(self, dimension, children_by_aggregators):
        """
            dimension: number of leaves
            children_by_aggregators: the number of leaves for each CI
        """
        nb_intermediate_nodes = dimension//children_by_aggregators + ((dimension%children_by_aggregators)!=0)
        starting_index_aggs = dimension
        hierarchy = {starting_index_aggs+agg:list(range(agg*children_by_aggregators,min((agg+1)*children_by_aggregators, dimension))) 
                     for agg in range(nb_intermediate_nodes)}
        hierarchy[-1] = list(hierarchy.keys())
        super(HCI2layers, self).__init__(hierarchy)

class HCIBalanced(HCI):
    """
        A tree-HCI with a single root and where all aggregators have the same number of leaves 
        (except the last one at each level)
    """
    def __init__(self, dimension, children_by_aggregators):
        """
            dimension: number of leaves
            children_by_aggregators: the number of leaves for each CI
        """
        current_layer = []
        next_layer = list(range(dimension))
        hierarchy = {}
        lay_i = 0
        while len(next_layer)>1:
            lay_i += 1
            current_layer = next_layer
            next_layer = []
            to_split = len(current_layer)
            next_layer_size = to_split//children_by_aggregators + ((to_split%children_by_aggregators)!=0)
            if next_layer_size==1:
                hierarchy[-1] = current_layer
                break
            reverse = 2*(lay_i%2)-1 #we alternate direction to avoid the last node always being in a small aggregator
            starting_index = max(current_layer)+1
            hierarchy = hierarchy|{starting_index+i: current_layer[::reverse][i*children_by_aggregators:(i+1)*children_by_aggregators] for i in range(next_layer_size)}
            next_layer = list(range(starting_index, starting_index+next_layer_size))
        super(HCIBalanced, self).__init__(hierarchy)
