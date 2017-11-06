""" Module that implements Subnetwork, the building block for a symflow
    network.
"""
import numpy as np
from collections import namedtuple
import tensorflow as tf
import preprocessing
from preprocessing import Dataset, RadParameters, AngParameters
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import network
import ml_util
import pickle

class Data():
    pass

class Subnet():
    """ Subnetwork that is associated with a chemical element or electron density
    Gaussians
    """

    seed = 42

    def __init__(self):
        self.species = None
        self.species_index = None
        self.n_copies = 0
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.name = None
        self.constructor = ml_util.fc_nn_g
        self.logits_name = None
        self.x_name = None
        self.y_name = None
        self.layers = [32] * 3
        self.targets = 1
        self.activations = [tf.nn.sigmoid] * 3
        self.features = 0

    def __add__(self, other):
        if not isinstance(other,Subnet):
            raise Exception("Incompatible data types")
        else:
            return network.Network([[self,other]])

    def __and__(self, other):
        if not isinstance(other,Subnet):
            raise Exception("Incompatible data types")
        else:
            return network.Network([[self],[other]])


    def get_feed(self, which, train_valid_split = 0.8, seed = None):
        """ Return a dictionary that can be used as a feed_dict in tensorflow

        Parameters:
        -----------
        which: {'train', 'valid', 'test'}, which part of the dataset is used
        train_valid_split: float; ratio of train and validation set size
        seed: int; seed parameter for the random shuffle algorithm, make

        Returns:
        --------
        dictionary
        """
        if seed == None:
            seed = Subnet.seed

        if train_valid_split == 1.0:
            shuffle = False
        else:
            shuffle = True


        if which == 'train' or which == 'valid':

            X_train = np.concatenate([self.X_train[i] for i in range(self.n_copies)],
                axis = 1)

            X_train, X_valid, y_train, y_valid = \
                train_test_split(X_train,self.y_train,
                                 test_size = 1 - train_valid_split,
                                 random_state = seed, shuffle = shuffle)
            X_train, X_valid = preprocessing.reshape_group(X_train, self.n_copies) , \
                               preprocessing.reshape_group(X_valid, self.n_copies)

            if which == 'train':
                return {self.x_name : X_train, self.y_name: y_train}
            else:
                return {self.x_name : X_valid, self.y_name : y_valid}

        elif which == 'test':

            return {self.x_name : self.X_test, self.y_name: self.y_test}



    def get_logits(self, i):
        """ Builds the subnetwork by defining logits and placeholders

        Parameters:
        -----------
        i: int; index to label datasets

        Returns:
        ---------
        logits, x, y: tensorflow tensors
        """

        with tf.variable_scope(self.name) as scope:
                        try:
                            logits,x,y_ = self.constructor(self, i,
                                np.mean(self.targets), np.std(self.targets))
                        except ValueError:
                            scope.reuse_variables()
                            logits,x,y_ = self.constructor(self, i,
                                np.mean(self.targets), np.std(self.targets))

        self.logits_name = logits.name
        self.x_name = x.name
        self.y_name = y_.name
        return logits, x, y_


    def save(self, path):
        """ Use pickle to save the subnet to path
        """

        with open(path,'wb') as file:
            pickle.dump(self,file)

    def load(self, path):
        """ Load subnet from path
        """

        with open(path, 'rb') as file:
            self = pickle.load(file)

    def add_datasets(self, X, targets,
        fraction = 1.0, test_size = 0.2, scale = True):
        """ Adds datasets to the subnetwork. First set in 'datasets' determines
        species that subnet is associated with

        Parameters:
        -----------
        X: dataset (defined in prepocessing.py); dataset that will be added
        to subnetwork for training and evaluation

        targets: (?,1) or (?) numpy array; target values for training and
            evaluation

        fraction: float, default = 1.0; fraction of data in X used in model
        (for large datasets)

        test_size: float < 1, default = 0.2; relative size of
                    the test set w.r.t the entire set

        scale: boolean, default = True; whether to scale the features.
               Default scaler is sklearn's MinMaxScaler but other scaler
               functions can be provided by setting self.scaler = custom_scaler

        Returns:
        -------
        None
        """

        # Find out if added dataset is compatible with contained ones
        if self.species != None and self.species_index != None:
            if self.species != X.species or \
                self.species != X.species_index:
                raise Exception("Dataset species does not equal subnet species")
        else:
            self.species = X.species
            self.species_index = X.species_index

        if not self.n_copies == 0:
            if self.n_copies != X.n:
                raise Exception("New dataset incompatible with contained one.")

        self.n_copies = X.n
        self.name = X.species


        # Preprocess
        features = preprocessing.reshape_dataset(X).data

        # Normalize
        features_flat = np.concatenate([features[i] for i in range(self.n_copies)],
            axis = 1)

        # Only use a fraction of the dataset
        if fraction < 1.0:
            y, _ = train_test_split(targets,
            test_size = 1-fraction, random_state = Subnet.seed, shuffle = True)
        else:
            y = targets

        # Split into training and test set
        if not test_size == 0.0:
            X_train, X_test, y_train, y_test = \
                train_test_split(features_flat, y,
                    test_size= test_size, random_state = Subnet.seed, shuffle = True)
        else:
            X_train = features_flat
            y_train = y
            X_test = np.copy(X_train)
            y_test = np.copy(y_train)

        if scale:
            if self.scaler == None:
                scaler = MinMaxScaler(feature_range = (-1,1), copy = False)
                scaler.fit(X_train.reshape(len(X_train) * self.n_copies,
                           int(X_train.shape[1]/self.n_copies)))
                print('No scaler provided, using MinMaxScaler fitted to ' +
                'training set')
                self.scaler = scaler
            else:
                scaler = self.scaler

        # Normalization parameters are determined by considering all copies
        # of one species
        X_train = preprocessing.reshape_group(X_train, self.n_copies)
        X_test = preprocessing.reshape_group(X_test, self.n_copies)

        if scale:
            for i in range(self.n_copies):
                X_train[i] = scaler.transform(X_train[i])
                X_test[i] = scaler.transform(X_test[i])

        if not isinstance(self.X_train, np.array):
            print('No dataset contained, initializing subnet with this dataset.')
            self.X_train = X_train
            self.X_test = X_test
            if y_train.ndim == 1:
                self.y_train = y_train.reshape(-1,1)
                self.y_test = y_test.reshape(-1,1)
            else:
                self.y_train = y_train
                self.y_test = y_test
            self.features = X_train.shape[2]
            self.targets = y_train.shape[1]
        else:
            if  X_train.shape[2] != self.X_train.shape[2] or \
                y_train.shape[1] != self.y_train.shape[1]:
                raise Exception("New dataset not compatible with contained one")

            self.X_train = np.concatenate([self.X_train,X_train], axis = 1)
            self.X_test = np.concatenate([self.X_test,X_test], axis = 1)
            self.y_train = np.concatenate([self.y_train,y_train])
            self.y_test = np.concatenate([self.y_test,y_test])
