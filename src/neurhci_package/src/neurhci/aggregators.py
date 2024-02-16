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

    def compute_mobius(self):
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

    def compute_shapley_values_global(self):
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

    def compute_local_shapley(self, x, y):
        """
            Computes the Shapley values for the input features for explaining the difference between two inputs
        """
        #TODO
        ...