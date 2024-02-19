#%%
from neurhci.uhci import UHCI

from neurhci.marginal_utilities import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#This file contains an example on how to build a model. Should be converted to a notebook.

#%% MODEL CREATION
"""
    This corresponds to a tree with root -1, and leaves 0,1,2,3 and 4 (since they have no child)
    5 is an intermediate node, which aggregates 3 and 4.
    The root -1 aggregates leaves 0,1,2 and 5.
"""
hierarchy={-1:[0,1,2,5], 5:[3,4]} 

"""
    We build our UHCI model with the hierarchy provided. 
    We precise that leaf 0 has a non-decreasing marginal utility, and leaf 2 has a non-increasing one.
    Leaves 1, 3 and 4 will have no marginal utility (i.e. an Indentity).

    The following blocks all yield the same output:

    option 1:
        hci = HCI(hierarchy)
        model(hci, types_of_leaves={0:NonDecreasing, 2:NonIncreasing})

    option 2:
        hci = HCI(hierarchy)
        marginal_utilities = MarginalUtilitiesLayer([NonDeacreasing, Identity, NonIncreasing, Identity, Identity])
        model(hci, marginal_utilities)
    
    option 3:
        marginal_utilities = MarginalUtilitiesLayer([NonDeacreasing, Identity, NonIncreasing, Identity, Identity])
        model(hierarchy, marginal_utilities)
    
    Providing an HCI or MarginalUtilitiesLayer is useful if some of the components are pre-trained or if they are shared
    with another model.
"""
model = UHCI(hierarchy=hierarchy, types_of_leaves={0:NonDecreasing, 2:NonIncreasing}, nb_sigmoids=15)

"""
    Please note that Identities have no learnable parameter and expect data in the unit interval (or [-0.5, 0.5] depending on 
    initialization parameters). See the marginal_utilities file for more details.

    Good practices include (any of these is enough):
        -using NonDecreasing/NonIncreasing instead of Identity/OppositeIdentity
        -using a normalization process/layer before the UHCI
        -using learnable sigmoids on the inputs before feeding them to the UHCI
"""
#%% SYNTHETIC DATA
def ground_truth_model(data):
    """
       A ground-truth model for our model to recover. We use it to generate the ground-truth labels.
       It is a HCI with same hierarchy as model, but described explicitely to avoid any ambiguity.
       Its marginal utilities are u_0(x_0)=x_0**3 and u_2(x_2) = 1-x**2. The rest are identity.
       It corresponds to the following function (noting ui = u_i(x_i) for simplicity):
            at the intermediate node 5:
                u5 = 0.4*u3 + 0.1*u4 + 0.5*min(u3, u4)
            at the root -1:
                u-1 = 0.2*u0 + 0.3*u2 + 0.2*u5 + 0.2*min(u0, u5) + 0.1*max(u1, u2) 
    """
    #applying marginal utilities:
    data_after_utilities = data.clone()
    data_after_utilities[:,0] = data_after_utilities[:,0]**3
    data_after_utilities[:,2] = 1-data_after_utilities[:,2]**2
    
    #aggregating at node 5:
    u5 = 0.4*data_after_utilities[:,3] + 0.1*data_after_utilities[:,4]
    u5 += torch.min(data_after_utilities[:,(3,4)], axis=1).values*0.5

    #aggregating at node -1
    gtd = 0.2*data_after_utilities[:,0] + 0.3*data_after_utilities[:,2] + 0.2*u5
    gtd += torch.min(torch.cat((data_after_utilities[:, 0:1], u5.unsqueeze(1)), axis=-1), axis=1).values*0.2
    gtd += torch.max(data_after_utilities[:, (1,2)], axis=1).values*0.1
    gtd = gtd.unsqueeze(1)
    return(data_after_utilities, gtd)

data_train = torch.rand((200,5)) #draw random data for training ; since the model is (very) small, we do not need a lot.
data_test = torch.rand((1000,5)) #draw random data for testing

train_after_utilities, labels_train = ground_truth_model(data_train) #compute the ground-truth labels
test_after_utilities, labels_test = ground_truth_model(data_test) #compute the ground-truth labels

#%% TRAINING
# Then we train using a basic torch process
optimizer = optim.Adam(model.parameters(), lr=0.02) # can be tuned for better results
criterion = nn.MSELoss()

for epoch in range(2000):
    optimizer.zero_grad()
    pred = model(data_train)
    training_loss = criterion(pred, labels_train)
    training_loss.backward(retain_graph=True)
    optimizer.step()
    if not epoch%100:
        with torch.no_grad():
            pred_test = model(data_test)
            testing_loss = criterion(pred_test, labels_test)
        print(f'Epoch {epoch}, training loss {training_loss}, testing loss {testing_loss}')
with torch.no_grad():
    pred_train = model(data_train)
    training_loss = criterion(pred_train, labels_train)
    pred_test = model(data_test)
    testing_loss = criterion(pred_test, labels_test)
print(f'Final training loss {training_loss}, Final testing loss {testing_loss}')

# Now our model should be trained. We check below if we learned the right values:
#%% PLOTTING THE MARGINAL UTILITIES
# We plot the learned marginal utilities against the testing data 
x = torch.linspace(0,1,100).unsqueeze(1)
gt_utilities = {0:lambda x:x**3, 2:lambda x:1-x*x}
for i in range(5):
    plt.subplot(2,3,i+1)
    ground_truth = lambda x:x
    if i in gt_utilities:
        ground_truth = gt_utilities[i]
    plt.plot(x.detach().numpy(), ground_truth(x).detach().numpy(), color='blue', marker='+', markevery=10, label='Ground Truth')
    plt.plot(x.detach().numpy(), model.marginal_utilities[i](x).detach().numpy(), color='red', marker='x', markevery=15, label="Learned")
plt.legend()
# %% ANALYZING THE AGGREGATORS
"""
We can check that the weights fit the ground truth's.

Recall that:
u5 = 0.4*u3 + 0.1*u4 + 0.5*min(u3, u4)

The weights are ordered such that (with n the size of the aggregator):
    -the nodes from 0 to n-1 weights correspond to the n leaves, in order
    -the nodes from n to n(n+1)/2-1 correspond to the min between pairs, with pairs
        following the lexicographic ordering: (0,1),(0,2),(...),(0,n),(1,2),(1,3)...,(n-1,n)
    -the nodes from n(n+1)/2 to n**2-1 correspond to max between pairs, ordered in the same way

We thus expect the weight vector for u5 to be:
[0.4, 0.1, 0.5, 0.]

Since these weights are not identifiable, we need to compute their (unique) equivalent canonical form (see thesis).

Note that the aggregators are accessed through the string version of their identifier (imposed by the nn.ModuleDict class).
"""
cweights5 = model.HCI.CIs['5'].canonical_weights()
print("--------------------------------")
print("For aggregator 5:")
print(f"Weight for u3          - ground truth 0.4 - learned value {cweights5[0,0]}")
print(f"Weight for u4          - ground truth 0.1 - learned value {cweights5[0,1]}")
print(f"Weight for min(u3, u4) - ground truth 0.5 - learned value {cweights5[0,2]}")
print(f"Weight for max(u3, u4) - ground truth 0.0 - learned value {cweights5[0,3]}")

"""
In the same way:

u-1 = 0.2*u0 + 0.3*u2 + 0.2*u5 + 0.2*min(u0, u5) + 0.1*max(u1, u2)
"""
print()
print("--------------------------------")
print("For aggregator -1:")
cweightsm1 = model.HCI.CIs['-1'].canonical_weights()
print(f"Weight for u0          - ground truth 0.2 - learned value {cweightsm1[0,0]}")
print(f"Weight for u1          - ground truth 0.0 - learned value {cweightsm1[0,1]}")
print(f"Weight for u2          - ground truth 0.3 - learned value {cweightsm1[0,2]}")
print(f"Weight for u5          - ground truth 0.2 - learned value {cweightsm1[0,3]}")
print(f"Weight for min(u0, u1) - ground truth 0.0 - learned value {cweightsm1[0,4]}")
print(f"Weight for min(u0, u2) - ground truth 0.0 - learned value {cweightsm1[0,5]}")
print(f"Weight for min(u0, u5) - ground truth 0.2 - learned value {cweightsm1[0,6]}")
print(f"Weight for min(u1, u2) - ground truth 0.0 - learned value {cweightsm1[0,7]}")
print(f"Weight for min(u1, x5) - ground truth 0.0 - learned value {cweightsm1[0,8]}")
print(f"Weight for min(u2, x5) - ground truth 0.0 - learned value {cweightsm1[0,9]}")
print(f"Weight for max(u0, x1) - ground truth 0.0 - learned value {cweightsm1[0,10]}")
print(f"Weight for max(u0, x2) - ground truth 0.0 - learned value {cweightsm1[0,11]}")
print(f"Weight for max(u0, x5) - ground truth 0.0 - learned value {cweightsm1[0,12]}")
print(f"Weight for max(u1, x2) - ground truth 0.1 - learned value {cweightsm1[0,13]}")
print(f"Weight for max(u1, x5) - ground truth 0.0 - learned value {cweightsm1[0,14]}")
print(f"Weight for max(u2, x5) - ground truth 0.0 - learned value {cweightsm1[0,15]}")

# %% MÖBIUS TRANSFORMS
"""
    The Mobius value of a given aggregator can be obtained through the mobius() method
    The ordering is the same as above: first the n singletons, then the n(n-1)/2 pairs in lexicographic order
"""
mob5 = model.HCI.CIs['5'].mobius()
print(f"""Möbius values for aggregator 5: {mob5}.
      The synergy between 3 and 4 is observed by the positive value
        of the term corresponding to their min: {mob5[-1]}.
      """)

mobm1 = model.HCI.CIs['-1'].mobius()
print(f"""Möbius values for aggregator -1: 
      {mobm1}.
    The synergy between 0 and 5 is observed by the positive value
        of the term corresponding to their pair: {mobm1[6]}.
    The redundancy between 1 and 2 is observed by the negative value
        of the term corresponding to their pair: {mobm1[7]}.
    """)
# %% EXPLANATIONS - LOCAL (SHAPLEY)
"""
    Finally, we show how to obtain the importance indices for the nodes 
        (i.e. Shapley and Winter values), in both a global and an instance-wise context.
    
    The global Shapley values give us the average local relative contribution
        of each child of a given aggregator wrt the value of said aggregator.
"""
shap5 = model.HCI.CIs['5'].shapley_values_global()
print("Children of 5 and their global Shapley values wrt 5:")
for c,s in zip(model.HCI.hierarchy[5], shap5):
    print(f"{c} has a relative global influence of {s}")
print()
shapm1 = model.HCI.CIs['-1'].shapley_values_global()
print("Children of -1 and their global Shapley values wrt -1:")
for c,s in zip(model.HCI.hierarchy[-1], shapm1):
    print(f"{c} has a relative global influence of {s}")
# %% EXPLANATIONS - WHOLE MODEL (WINTER)
"""
    To obtain the global influence of a leaf wrt the root (or, if needed, of any node wrt to any of its ancestors), 
        we compute the global Winter value (i.e. the generalization of the global Shapley values to a structured model).

    Note that the model induced by the descendents of the ancestor node MUST be a tree.
"""
influence_of_leaves = model.HCI.winter_values_global() # If no argument is passed, the root is explained.
for l,w in influence_of_leaves.items():
    print(f"{l} has a relative global influence of {w} on the root -1.")
print("Note that each non-leaf node's influence is the sum of the influence of its children.")

"""
    To explain for a specific node, we use the "starting_node" argument. Note that since it is a relative influence,
    the starting_node's influence on itself will always be 100%. This is akin to only observing the HCI induced by
    the starting node's descendents, ignoring all other parts of the bigger model.
"""
print()
influence_on_5 = model.HCI.winter_values_global(starting_node=5)
for l,w in influence_on_5.items():
    print(f"{l} has a relative global influence of {w} on the node 5.")

#%% EXPLANATION - INSTANCE-WISE
"""
    We now demonstrate how to obtain the instance-wise influence of a node on one of its ancestor.
    That is, rather than "how important is the node on average", we focus on "how important is that node
    to explain the difference between two predictions".

    This contribution can be positive (the node aims at increasing the difference) or negative.
    0. means that the node does not contribute to the difference.

    To do that, we use the Winter values:
"""
X = torch.rand(3, 5)
Y = torch.rand(3, 5)
# The comparisons will be made linewise, i.e. x[0] vs y[0], x[1] vs y[1]...
# In this case, there will be three comparisons.

winter_values = model.winter_values(X, Y) #since the starting node is not given, the root will be used.
for i,(x,y) in enumerate(zip(X,Y)):
    predx, predy = model(x.unsqueeze(0)), model(y.unsqueeze(0))
    print(f"x = {x}, with prediction {predx.item()}")
    print(f"y = {y}, with prediction {predy.item()}")
    print(f"The difference between predictions is {predy.item()-predx.item()}")
    print(f"The share of each leaf in the difference in prediction is:")
    for k,v in winter_values.items():
        print(f"    Leaf {k} contributes for {v[i].item()} to the difference")
    print()
#Note that the sum of contibutions of all leaves sums up to the difference. This is the efficiency property.

"""
    The Winter value of a single node wrt another can also be computed.
    Note that this is useful to look at internal relations among nodes.
    Note that the importance of a node i wrt to another node j is always 
        equal to the sum of the importances of the leaves descendent of 
        node i wrt node j 
"""
winter_value_5 = model.winter_value_single_node(X, Y, 5, -1) #importance of node 5 wrt node -1
winter_values_3p4 = winter_values[3]+winter_values[4]
for i in range(3):
    print(f'Contributions of node 5 to node -1 for example {i+1}:')
    print(winter_value_5[i].item())
    print(f'Sum of contributions of nodes 3 and 4 to node -1 for example {i+1}:')
    print(winter_values_3p4[i].item())