import torch
import torch.nn as nn
from .hierarchical import HCI
from .marginal_utilities import MarginalUtilitiesLayer, Identity

class UHCI(nn.Module):
    """
        A utilitaristic hierarchical Choquet integral, combination of marginal utilities and a HCI.
    """

    def __init__(self, **kwargs):
        """
            possible arguments:
                hierarchy: dict representation, same as the one used for the HCI
                hci: an HCI object (overrides hierarchy if both are given)
                types_of_leaves: dict representation, same as the one used for a MarginalUtilitiesLayer
                nb_sigmoids: int, number of sigmoids to be used in the marginal utilities if applicable
                marginal_utilities: a MarginalUtilitiesLayer object (overrides types_of_leaves and nb_sigmoids if provided)
        """
        super(UHCI, self).__init__()
        nb_sigmoids = kwargs.get('nb_sigmoids', 100)
        try:
            self.HCI = kwargs['hci'] if 'hci' in kwargs else HCI(kwargs['hierarchy'])
        except:
            raise TypeError("Neither an HCI nor a hierarchy were given")
        if ('marginal_utilities' in kwargs)|('types_of_leaves' in kwargs):
            if 'marginal_utilities' in kwargs:
                self.marginal_utilities = kwargs['marginal_utilities']  
            else:
                types_of_leaves = kwargs['types_of_leaves']
                self.marginal_utilities = MarginalUtilitiesLayer(self.HCI.leaves, types_of_leaves, nb_sigmoids)
        else:
            types_of_leaves = {i:Identity for i in self.HCI.leaves}
            self.marginal_utilities = MarginalUtilitiesLayer(self.HCI.leaves, types_of_leaves, nb_sigmoids)
        assert len(self.HCI.leaves)==len(self.marginal_utilities), "Not the same number of leaves and marginal utilities"

    def forward(self, x):
        x = self.marginal_utilities(x)
        x = self.HCI(x)
        return(x)

    def winter_value_single_node(self, x, y, explained_node, starting_node=None):
        """
            x,y: two tensors of size (m, dimension)
            starting_node: the node whose difference in value must be explained (by default, the root of the tree)
            explained_node: the node (feature) whose contribution to the difference must be computed

            For each of the m lines of x and y, computes the value of the contribution of the explained_node to the
                difference in value at starting_node between self.forward(x) and self.forward(y)
                The output has shape (m, 1).

            Note that the structure induced by the descendants of the starting_node
                needs to be a tree.
        """
        with torch.no_grad():
            ux, uy = self.marginal_utilities(x), self.marginal_utilities(y)
        return(self.HCI.winter_value_single_node(ux, uy, explained_node, starting_node))

    def winter_values(self, x, y, starting_node=None):
        """
            Computes the Winter values for all leaves.
        """
        with torch.no_grad():
            ux, uy = self.marginal_utilities(x), self.marginal_utilities(y)
        return(self.HCI.winter_values(ux, uy, starting_node))