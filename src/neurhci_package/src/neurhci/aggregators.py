import itertools
from math import factorial

import torch
import torch.nn as nn
import torch.nn.functional as F

class CI2Add(nn.Module):
    """
        A module representing a 2-additive Choquet integral.

        Input is a tensor of shape (m, d), with m the batch size and d the dimension.
        Output is a tensor of shape (m, 1)
    """
    def __init__(self, dimension):
        """
            dimension: the number of aggregated features
        """
        super(CI2Add, self).__init__()
        self.dim = dimension
        self.preweight = nn.Parameter(torch.randn((1,self.dim**2)))
        self.update_weight()
        self.output = 0.
        self.input = torch.zeros(self.dim)
    
    def update_weight(self):
        self.weight = F.softmax(self.preweight, dim=1)

    def forward(self, x):
        self.input = x
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        pairs_indices = torch.combinations(torch.arange(0, x.shape[1]), 2)
        pairs = x[:, pairs_indices]
        x_completed = torch.cat([x, torch.min(pairs, dim=2).values, torch.max(pairs, dim=2).values], dim=1)
        self.update_weight()
        self.output = F.linear(x_completed, self.weight)
        return(self.output)

    def force_weights(self, w):
        """
            Impose the weights values
        """
        with torch.no_grad():
            w = torch.clamp(w, 1e-9)
            w = w.reshape(1, self.dim**2)
            self.preweight = nn.Parameter(torch.log(w))
            self.update_weight()

    def canonical_weights(self):
        """
            Returns the canonical variant of the weights (i.e. the equivalent weight vector such that
            for any pair of criteria, at least one weight among w_{ij}^min/w_{ij}^max is 0)
        """
        canonical = torch.zeros_like(self.weight)
        shift = self.dim*(self.dim-1)//2
        canonical = 1.*self.weight
        current_element = self.dim
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                diff = min(self.weight[0,current_element],self.weight[0,current_element+shift])
                canonical[0,current_element] -= diff
                canonical[0,current_element+shift] -= diff
                canonical[0,i] += diff
                canonical[0,j] += diff
                current_element += 1
        return(canonical)

    def mobius(self):
        """
            Computes the Möbius transform of the fuzzy measure 
        """
        self.update_weight()
        with torch.no_grad():
            w = self.weight[0]
            mobius = torch.zeros(self.dim*(self.dim+1)//2)
            mobius[:self.dim] += w[:self.dim]
            current_element = self.dim
            for i in range(self.dim):
                for j in range(i+1, self.dim):
                    weight_min = w[current_element]
                    weight_max = w[current_element + (self.dim*(self.dim-1))//2]
                    mobius[current_element] = weight_min-weight_max
                    mobius[i] += weight_max
                    mobius[j] += weight_max
                    current_element += 1
        return(mobius)

    def shapley_values_global(self):
        """
            Computes the global Shapley values of the input features, i.e. their average contribution to the model
        """
        with torch.no_grad():
            w = self.weight[0]
            shapley = torch.zeros(self.dim)
            shapley[:self.dim] += w[:self.dim]
            current_node = self.dim
            for i in range(self.dim):
                for j in range(i+1, self.dim):
                    weight_min = w[current_node]
                    weight_max = w[current_node + (self.dim*(self.dim-1))//2]
                    shapley[i] += (weight_max + weight_min)/2
                    shapley[j] += (weight_max + weight_min)/2
                    current_node += 1
        return(shapley)

    def shapley_value_single_node(self, x, y, i):
        """
            x: tensor of shape b x self.dim
            y: tensor of shape b x self.dim
            i: integer, index of child to explain

            Computes the Shapley values for the ith input feature for explaining the difference between two inputs.
        """
        with torch.no_grad():
            shap = torch.zeros(x.shape[0], 1)
            divisor = 0.
            for pos_in_permutation in range(self.dim):
                #We do not explicitely use all permutations, sice many are equivalent for our purpose.
                #Thus, we only consider splits, and reajust by the number of underlying permutations represented by each split.
                other_children = [c for c in range(self.dim) if c!=i]
                all_splits = itertools.combinations(other_children, pos_in_permutation)
                for split_children in all_splits:
                    after_explained = list(split_children) #these  elements will be y
                    nb_permutations = factorial(pos_in_permutation)*factorial(self.dim-pos_in_permutation-1)
                    first_half = x*1
                    first_half[:, after_explained] = y[:, after_explained]
                    second_half = first_half*1
                    second_half[:, i] = y[:, i]
                    compounds_inputs = torch.cat([first_half, second_half])
                    compounds_outputs = self.forward(compounds_inputs)
                    deltas = compounds_outputs[x.shape[0]:]-compounds_outputs[:x.shape[0]]
                    shap += deltas*nb_permutations
                    divisor += nb_permutations
            shap /= divisor
        return(shap)

    def shapley_values(self, x, y):
        """
            x: tensor of shape b x self.dim
            y: tensor of shape b x self.dim

            Computes the Shapley values for input feature for explaining the difference between two inputs.
        """
        shap = torch.cat([self.local_shapley_value(x, y, i) for i in range(self.dim)], dim=1)
        return(shap)

class CI01FromAntichain(nn.Module):
    """
        A Choquet Integral with a 0-1 Fuzzy Measure defined as its associated antichain.

        An antichain is a tuple of tuples of indices. The associated CI is provably a max(min)
    """
    def __init__(self, antichain):
        super(CI01FromAntichain, self).__init__()
        self.antichain = antichain
        self.output = 0.

    def forward(self, x):
        self.input = x
        get_mins = torch.cat([torch.min(x[:,group], dim=1).values.unsqueeze(1) for group in self.antichain], dim=1)
        self.output = torch.max(get_mins, dim=1).values.unsqueeze(1)
        return(self.output)

class CI3addFrom01(nn.Module):
    # could be optimized through tensor manipulation
    def __init__(self, dimension):
        super(CI3addFrom01, self).__init__()
        self.dim = dimension
        all_doubles = itertools.combinations(range(self.dim), 2)
        all_triples = itertools.combinations(range(self.dim), 3)
        self.zero_one_CIs = torch.nn.ModuleList({})
        for i in range(self.dim):
            self.zero_one_CIs.append(CI01FromAntichain(((i,),)))
        for i,j in all_doubles:
            self.zero_one_CIs.append(CI01FromAntichain(((i,j),)))
            self.zero_one_CIs.append(CI01FromAntichain(((i,),(j,))))
        for i,j,k in all_triples:
            ac = [
                ((i,j,k),),
                ((i,),(j,),(k,)),
                ((i,),(j,k)),
                ((j,),(i,k)),
                ((k,),(i,j)),
                ((i,j),(j,k)),
                ((i,k),(j,k)),
                ((i,j),(i,k)),
            ]
            for a in ac:
                self.zero_one_CIs.append(CI01FromAntichain(a))
        self.preweight = nn.Parameter(torch.randn((1,len(self.zero_one_CIs))))

    def update_weight(self):
        self.weight = F.softmax(self.preweight, dim=1)

    def forward(self, x):
        self.input = x
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        x_completed = torch.cat([zoci(x) for zoci in self.zero_one_CIs], dim=1)
        self.update_weight()
        self.output = F.linear(x_completed, self.weight)
        return(self.output)
