""" Module that contains functions to preprocess quantumm chemical data such as
atomic positions and/or positions and width of Gaussians that are fitted
to the charge density. Preprocessed datasets are then used in subnet and network,
for a Neural Network similar to that proposed by Behler et al.
"""

import numpy as np
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Dataset containing the coordinates and width ( = 1 for atoms) for
# Gaussians and Atoms
Dataset = namedtuple("Dataset", "data species species_index n")

def reshape_dataset(d):
    """Reshape data from format (n_samples * n_copies, n_features)
    into format (n_copies, n_samples, n_features)
    """
    d = d._replace(data = d.data.reshape(-1,d.n,d.data.shape[1]))
    d.swapaxes(0,1)

    return d

def reshape_group(x, n):
    """Reshape data from flat format (n_samples, n_copies * n_features)
    into format (n_copies, n_samples, n_features) needed by tensorflow
    """

    n0 = x.shape[0]
    n1 = int(x.shape[1]/n)
    x = x.T.reshape(n,n1,n0).swapaxes(1,2)

    return x

def scale_together(subnets):
    """ Scale data in given subnets by combining the data ranges
        (Should always be performed if subnets share weights)
    """

    for s in subnets:
        if not s.species == subnets[0].species:
            print('Warning, subnets do not contain the same species. Proceeding...')

    if not isinstance(subnets ,list):
        Exception('Input must be a list of subnets')

    all_data = np.zeros([0,s.features])
    for i, s in enumerate(subnets):
        subnets[i].X_train= subnets[i].X_train.reshape(-1, subnets[i].features)
        subnets[i].X_train = subnets[i].scaler.inverse_transform(subnets[i].X_train)
        subnets[i].X_test= subnets[i].X_test.reshape(-1, subnets[i].features)
        subnets[i].X_test = subnets[i].scaler.inverse_transform(subnets[i].X_test)
        all_data = np.concatenate([all_data,subnets[i].X_train], axis=0)

    subnets[0].scaler.fit(all_data)

    for i,s in enumerate(subnets):
        subnets[i].scaler = subnets[0].scaler
        subnets[i].X_train = subnets[i].scaler.transform(subnets[i].X_train)
        subnets[i].X_train = subnets[i].X_train.reshape([-1, subnets[i].features*s.n_copies])
        subnets[i].X_train = reshape_group(subnets[i].X_train, subnets[i].n_copies)
        subnets[i].X_test = subnets[i].scaler.transform(subnets[i].X_test)
        subnets[i].X_test = subnets[i].X_test.reshape([-1, subnets[i].features*subnets[i].n_copies])
        subnets[i].X_test = reshape_group(subnets[i].X_test, subnets[i].n_copies)

    return subnets
