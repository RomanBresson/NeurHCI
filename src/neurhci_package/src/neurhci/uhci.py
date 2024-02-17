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
                type_of_leaves: dict representation, same as the one used for a MarginalUtilitiesLayer
                nb_sigmoids: int, number of sigmoids to be used in the marginal utilities if applicable
                marginal_utilities: a MarginalUtilitiesLayer object (overrides type_of_leaves and nb_sigmoids if provided)
        """
        super(UHCI, self).__init__()
        nb_sigmoids = kwargs.get('nb_sigmoids', 100)
        try:
            self.HCI = kwargs['hci'] if 'hci' in kwargs else HCI(kwargs['hierarchy'])
        except:
            raise TypeError("Neither an HCI nor a hierarchy were given")
        try:
            if 'marginal_utilities' in kwargs:
                self.marginal_utilities = kwargs['marginal_utilities']  
            else:
                type_of_leaves = kwargs['types_of_leaves']
        except:
            type_of_leaves = {i:Identity for i in self.HCI.leaves}
        self.marginal_utilities = MarginalUtilitiesLayer(self.HCI.leaves, type_of_leaves, nb_sigmoids)
        assert len(self.HCI.leaves)==len(self.marginal_utilities), "Not the same number of leaves and marginal utilities"

    def forward(self, x):
        x = self.marginal_utilities(x)
        x = self.HCI(x)
        return(x)