# Neur-HCI

This repository is the official implementation of Neur-HCI as introduced in our [paper](https://www.ijcai.org/proceedings/2020/0275.pdf) and completed in our [thesis](https://theses.hal.science/tel-03596964). It consists in neural network architectures made to learn utilitaristic hierarchical Choquet integrals (UHCI), a class of models used in multicriteria decision aiding.

This is a reimplementation of the code used in those papers, and not the original code.

For each class, a representation theorem is provided and proven in the thesis, such that the search space exactly coincides with the sought class. All theoretical properties applicable to Choquet integrals are thus formally valid by design.

## Requirements

The code was written with Python 3.11, Python 3.8 or above should work.
The requirements are listed in src/neurhci_package/pyproject.toml
Current requirements (with versions used for development):
torch==2.2.0

## Documentation

### Tutorial

A jupyter notebook is provided in src/example_scripts/tutorial.ipynb. It should give one an overview of the classes, the methods, and overall how to:
-build a model
-train a model
-extract explanations (Shapley/Winter values) from the trained model.

### Installation

Get the code with:
```
git clone https://github.com/RomanBresson/NeurHCI.git
```

The code is packaged to be installed through pip with:
```
cd NeurHCI/src/neurhci_package
pip install .
```

Once installed, the package can be imported like any other pip package:
```
import neurhci
```

### Submodules

The classes detailed here, module by module, either inherit from PyTorch's ``nn.Module``, and can thus be used like any basic module. All these modules are proven to have the monotonicity required by the model. All models (except those who are explicitely excluded below) respect the normalization constraints of the model. See the sources for clearer understanding of the underlying theory.

* ``marginal_utilities``: classes implementing marginal utilities. It is to be noted that many properties of the model only hold if the outputs of the marginal utility is in the unit interval. Their default initialization is best for inputs who are also in the unit interval. When the data is normalized around 0, passing ``centered=True`` to the constructors makes it the same function applied to x+0.5, which should yield a better initialization.
  * ``Identity()``: $u(x) = x$ (no learnable parameter, does not respect normalization if the inputs are not normalized in the unit interval)
  * ``OppositeIdentity()``: $u(x) = 1-x$ (no learnable parameter, does not respect normalization if the inputs are not normalized in the unit interval)
  * ``IdentityClipped``/``OppositeIdentityClipped``: variants of Identity and OppositeIdentity whose outputs are clipped in the unit interval to ensure normalization.
  * ``NonDecreasing(nb_sigmoids)``: $u(x) = \sum\limits_{i=1}^p w_i\sigma(\eta_i(x-\beta_i))$ with $p$ the number of sigmoids, $\eta,~\beta,~w$ being learned, and $\sigma$ being a logistic sigmoid. Can represent any non-decreasing function with image in the unit interval (and only those).
  * ``NonIncreasing(nb_sigmoids)``: $u(x) = 1-v(x)$ with $v$ a ``NonDecreasing``.
  * ``Unconstrained(nb_layers, width)``: a simple MLP with 1d input, 1d output, and ``nb_layers`` fully connected hidden layers, each with ``width`` neurons. DOES NOT FIT ANY CONSTRAINTS OF NORMALIZATION OF MONOTONICITY, it is simply here if required for some specific cases.
  * ``MarginalUtilitiesLayer(list_of_leaves, types_of_leaves, nb_sigmoids)``: a list of marginal utilities ${u_1,...,u_n}$ where $u_i$ corresponds to ``list_of_leaves[i]`` and has type ``types_of_leaves[list_of_leaves[i]]``. Any non-given type will be replaced by an ``Identity``.
* ``aggregators``: classes implementing Choquet integral-based aggregators:
  * ``CI2Add``: The $2$-additive Choquet integral, with ``dimension`` inputs.
* ``hierarchical``: class of hierarchical Choquet integral, i.e. a multi-step aggregator which aggregates the inputs successively following a directed-acyclic graph structure. Contains the following classes:
  * ``HCI(hierarchy)``: builds a HCI with the structure passed as argument. ``hierarchy`` is a dict of {(int) key: (list of int) value} where key is the id of a node, and value is the list of this node's children. Details and examples can be found below.
  * ``HCI2layers(dimension, children_by_aggregators)``: A tree-HCI with a single root single intermediate layer, and ``dimension leaves``. Each node in the intermediate layer aggregates ``children_by_aggregators`` leaves (or fewer for the last one). The root aggregates all intermediate nodes
  * ``HCIBalanced(dimension, children_by_aggregators)``: A tree-HCI with a single root, ``dimension`` leaves, where all aggregators have the same number ``children_by_aggregators`` of leaves (except the first/last one at each level)
* ``uhci``:
  * ``UHCI(**kwargs)``: A utilitaristic hierarchical Choquet integral, combination of marginal utilities and a HCI. Can be initialized from an existing ``HCI`` or a hierarchy, and from an existing list of marginal utilities or a ``MarginalUtilitiesLayer``.

### To do

The classes described above are just a part of the models described in the thesis. The rest will be implemented at a later date, such as:
* 3-additive Choquet integrals based on 0-1 measures
* General Choquet integrals
* Bitonic marginal utilities (single peaked/single valleyed)
* Marginal utilities selectors, to automatically select the best monotonicity type

### Cite
When using this package, please cite one of our papers:

If using only 2-additive Choquet integrals and/or monotonic marginal utilities:
```
@inproceedings{ijcai2020p0275,
  title     = {Neural Representation and Learning of Hierarchical 2-additive Choquet Integrals},
  author    = {Bresson, Roman and Cohen, Johanne and Hüllermeier, Eyke and Labreuche, Christophe and Sebag, Michèle},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Christian Bessiere},
  pages     = {1984--1991},
  year      = {2020},
  month     = {7},
  note      = {Main track},
  doi       = {10.24963/ijcai.2020/275},
  url       = {https://doi.org/10.24963/ijcai.2020/275},
}
```

If using any other class:
```
@phdthesis{bresson:tel-03596964,
  TITLE = {{Neural learning and validation of hierarchical multi-criteria decision aiding models with interacting criteria}},
  AUTHOR = {Bresson, Roman},
  URL = {https://theses.hal.science/tel-03596964},
  NUMBER = {2022UPASG008},
  SCHOOL = {{Universit{\'e} Paris-Saclay}},
  YEAR = {2022},
  MONTH = Feb,
  KEYWORDS = {Multi-Criteria Decision Aiding ; Choquet Integral ; Machine Learning ; Trustable AI ; Aide {\`a} la D{\'e}cision Multicrit{\`e}re ; Int{\'e}grale de Choquet ; Apprentissage Automatique ; IA de confiance},
  TYPE = {Theses},
  PDF = {https://theses.hal.science/tel-03596964/file/107767_BRESSON_2022_archivage.pdf},
  HAL_ID = {tel-03596964},
  HAL_VERSION = {v1},
}
```
