import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self, *args):
        super(Identity, self).__init__()
    
    def forward(self, x):
        self.output = x
        return(self.output)

class OppositeIdentity(nn.Module):
    def __init__(self, *args):
        super(OppositeIdentity, self).__init__()
    
    def forward(self, x):
        self.output = 1-x
        return(self.output)

class NonDecreasing(nn.Module):
    def __init__(self, nb_sigmoids=100):
        super(NonDecreasing, self).__init__()
        self.preweight = torch.nn.Parameter(torch.randn(1, nb_sigmoids))
        self.weight = F.softmax(self.preweight, dim=-1)
        self.pre_precision = torch.nn.Parameter(torch.abs(torch.randn(1, nb_sigmoids)*nb_sigmoids))
        self.precision = F.softplus(self.pre_precision)
        self.bias = torch.nn.Parameter(torch.rand(1, nb_sigmoids))

    def forward(self, x):
        self.precision = F.softplus(self.pre_precision)
        self.weight = F.softmax(self.preweight, dim=-1)
        x = x-self.bias
        x = self.precision*x
        x = F.sigmoid(x)
        x = x@self.weight.T
        self.output = x
        return(self.output)

class NonIncreasing(NonDecreasing):
    def __init__(self, nb_sigmoids=100):
        super(NonIncreasing, self).__init__(nb_sigmoids)
    
    def forward(self, x):
        x = super(NonIncreasing, self).forward(x)
        return(1-x)

class Unconstrained(nn.Module):
    """
        Basically a 1d to 1d MLP
    """
    def __init__(self, nb_layers=2, width=100):
        super(Unconstrained, self).__init__()
        modules = [nn.Linear(1, width), nn.ReLU()]
        for _ in range(nb_layers-2):
            modules.append(nn.Linear(width, width))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(width, 1))
        self.linears = nn.Sequential(*modules)

    def forward(self, x):
        self.output = self.linears(x)
        return(self.output)

node_codes = {
    "UId" : Identity,
    "UNId": OppositeIdentity,
    "UND" : NonDecreasing,
    "UNI" : NonIncreasing,
    "UUn" : Unconstrained
}

class MarginalUtilitiesLayer(nn.ModuleList):
    def __init__(self, list_of_leaves, types_of_nodes, nb_sigmoids):
        super(MarginalUtilitiesLayer, self).__init__()
        if types_of_nodes is None:
            types_of_nodes = {}
        for l in list_of_leaves:
            if l not in types_of_nodes:
                types_of_nodes[l] = "UId"
            self.append(node_codes[types_of_nodes[l]](nb_sigmoids))

    def forward(self, x):
        x = torch.cat([ui(x[:,i:i+1]) for i,ui in enumerate(self)], axis=-1)
        return(x)