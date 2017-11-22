import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from . import ml_util as ml
import pickle
# Set for additional output
debug = False

class _Node():
    """ Parent class, should not be instantiated
    """

    def __init__(self, name = 'default'):
        self.sends_to = []
        self.receives_from = []
        self.tensor_name = None
        self.name = name

    def __rshift__(self, other):
        """ Send to other """
        self.update_sends_to(other)
        return other

    def __lshift__(self, other):
        """ Other sends_to self"""
        other.update_sends_to(self)
        return other

    def update_sends_to(self, to):
        """ update the sends_to attribute making sure
        no redundancies are added
        """
        if not (to in self.sends_to):
            self.sends_to.append(to)

    def sends_to_names(self):
        """ Return a list of names of all the nodes self sends data to
        """
        names = []
        for to in self.sends_to:
            names.append(to.name)
        return names

    def findall_forward(self):
        """ Recursive function used by findall_nodes (graph.py) to find
        all nodes in graph. Determines all the nodes self is sending data
        to"""

        nodes = [self]
        for n in self.sends_to:
            if isinstance(n,Datanode):
                continue
            nodes += n.findall_forward()
        return nodes

    def op(self, x):
        """ Whatever operation this node should perform. Has to return
        a tensorflow tensor

        Parameters
        ---------
        x : Input tensor

        Returns
        --------
        tensorflow tensor

        """
        raise NotImplementedError("op not implemented for this child \
            class")

    def get_tensors(self, x):
        """ Builds the subnetwork by defining logits

        Parameters:
        -----------
        x: tf.placeholder for input data

        Returns:
        ---------
        tensor, None: tf.tensor for output data, the None is needed so that
        the output signature is the same as for get_tensors in Datanode

        """
        if self.tensor_name == None:
            tensor = self.op(x)
            self.tensor_name = tensor.name
            self.tensor = tensor

        return self.tensor, None

    def get_n_output(self):
        raise NotImplementedError("get_n_output not implemented for this child \
            class")

    def connect_backwards(self):
        """ Builds the neural net tensors recursively by starting at the nodes
        connected to the target Datanode and working its way back through all nodes.
        This has to be done recursively as the input of Subnetnode (x in
        get_tensors(x)) is determined by the output of all the Nodes it is
        connected to
        """

        input_tensors = []

        #Automatically determine size of input tensor for this subnet
        n_features = 0

        for rf in self.receives_from:
            if debug: print('Connecting {} to {}'.format(self.name, rf.name))
            input_tensors.append(rf.get_tensors(rf.connect_backwards())[0])
            n_features += rf.get_n_output()

        self.features = n_features
        if len(input_tensors) > 1:
            return tf.concat(input_tensors, axis = 1)
            if debug: print('(Receives from multiple tensors)')
        else:
            return input_tensors[0]

    def set_prefactors(self):
        pass

class Datanode(_Node):

    seed = 42
    index = 0

    def __init__(self, X=None, y=None, scaler=None, name='Datanode',
                 test_size=0.2):

        self.scaler = scaler



        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.x_name = None
        self.y_name = None
        self.x = None
        self.y_ = None
        self.test_size = test_size
        super().__init__()

        self.name = name
        if self.name == 'Datanode':
            if debug: print('Warning, name not set. Using default: "Datanode" ')

        # Keep track of Datanode instances and index them
        self.index = Datanode.index
        Datanode.index += 1

        if isinstance(X,np.ndarray) and isinstance(y, np.ndarray):
            self.add_dataset(X,y)
        elif isinstance(X,np.ndarray):
            self.add_dataset(X)


    def add_dataset(self, X, y = 0, test_size = -1, scale = True):

        """ Adds dataset to Datanode. First added set determines
        species that node is associated with

        Parameters:
        -----------
        X: numpy array (defined in prepocessing.py); dataset that will be added
        to datanode for training and evaluation

        y: (?,1) or (?) numpy array; target values for training and
            evaluation

        test_size: float < 1, default = 0.2; relative size of
                    the test set w.r.t the entire set

        scale: boolean, default = True; whether to scale the features.
               Default scaler is sklearn's MinMaxScaler but other scaler
               functions can be provided by setting self.scaler = custom_scaler

        Returns:
        -------
        None
        """
        if test_size == -1: test_size = self.test_size
        # If dataset has no targets
        if not isinstance(y, np.ndarray):
            y = np.zeros([X.shape[0],1])

        # Split into training and test set
        if not test_size == 0.0:
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y,
                    test_size= test_size, random_state = Subnetnode.seed, shuffle = True)
        else:
            X_train = X
            y_train = y
            X_test = np.copy(X_train)
            y_test = np.copy(y_train)

        if scale:
            if self.scaler == None:
                scaler = MinMaxScaler(feature_range = (-1,1), copy = False)
                scaler.fit(X_train)
                if debug: print('No scaler provided, using MinMaxScaler fitted to ' +
                'training set')
                self.scaler = scaler
            else:
                scaler = self.scaler

            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        if not isinstance(self.X_train, np.ndarray):
            if debug: print('No dataset contained yet, initializing node with this dataset.')
            self.features = X_train.shape[1]
            self.X_train = np.zeros([0, self.features])
            self.X_test = np.zeros([0, self.features])

            if y_train.ndim == 1:
                y_train = y_train.reshape(-1,1)
                y_test = y_test.reshape(-1,1)

            self.targets = y_train.shape[1]
            self.y_train = np.zeros([0,self.targets])
            self.y_test = np.zeros([0,self.targets])

        else:
            if  X_train.shape[2] != self.X_train.shape[2] or \
                y_train.shape[1] != self.y_train.shape[1]:
                raise Exception("New dataset not compatible with contained one")

        self.X_train = np.concatenate([self.X_train,X_train])
        self.X_test = np.concatenate([self.X_test,X_test])
        self.y_train = np.concatenate([self.y_train,y_train])
        self.y_test = np.concatenate([self.y_test,y_test])



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
            seed = Subnetnode.seed

        if train_valid_split == 1.0:
            shuffle = False
        else:
            shuffle = True


        if which == 'train' or which == 'valid':

            X_train, X_valid, y_train, y_valid = \
                train_test_split(self.X_train, self.y_train,
                                 test_size = 1 - train_valid_split,
                                 random_state = seed, shuffle = shuffle)
            if which == 'train':
                return {self.x_name : X_train, self.y_name: y_train}
            else:
                return {self.x_name : X_valid, self.y_name : y_valid}

        elif which == 'test':

            return {self.x_name : self.X_test, self.y_name: self.y_test}

    def get_tensors(self, _ = None): # _ is dummy to make signature the
                                     # same as in Subsetnode
        """ Builds the Datanode by defining tf.placeholders for
            input data and targets

        Parameters:
        -----------
        None

        Returns:
        ---------
        x, y: tensorflow tensors
        """

        i = self.index
        features = self.X_train.shape[1]
        targets = self.y_train.shape[1]

        if self.x == None and self.y_ == None:
            x = tf.placeholder(tf.float32,[None,features],'x' + str(i))
            y_ = tf.placeholder(tf.float32,[None, targets], 'y_' + str(i))
            self.x = x
            self.y_ = y_
        else:
            x = self.x
            y_ = self.y_

        if self.x_name == None:
            self.x_name = x.name
            self.y_name = y_.name
        elif (x.name != self.x_name or y_.name != self.y_name):
            raise Exception('Placeholder naming conflict')

        return x, y_

    def connect_backwards(self): #Needed to end recursion on Datanode
        return None

    def get_n_output(self):
        return self.X_train.shape[1]

    def set_prefactors(self):
        pass

class Subnetnode(_Node):

    seed = 42

    def __init__(self, name, targets, prefactor = None):

        self.constructor = ml.fc_nn
        self.logits_name = None
        self.layers = [32] * 3
        self.targets = targets
        self.activations = [tf.nn.sigmoid] * 3
        self.features = 0
        self.logits = None

        if prefactor == None:
            # prefactor = [1]*self.targets
            prefactor = 1

        self.prefactor = prefactor

        self.prefactor_tensor = None
        super().__init__()
        self.name = name

    def set_prefactors(self): #TODO
        pass
        # if self.prefactor_tensor == None:
        #     self.prefactor_tensor = tf.constant(self.prefactor,dtype=tf.float32,
        #         shape=[1,len(self.prefactor)])



    def get_tensors(self, x):
        """ Builds the subnetwork by defining logits

        Parameters:
        -----------
        x: tf.placeholder for input data

        Returns:
        ---------
        logits, None: tf.placeholder for output data, the None is needed so that
        the output signature is the same as for get_tensors in Datanode

        """
        if self.logits_name == None:

            with tf.variable_scope(self.name) as scope:
                try:
                    logits = self.constructor(self, x)
                except ValueError:
                    if debug : print('Sharing variables')
                    scope.reuse_variables()
                    logits = self.constructor(self, x)

            self.logits_name = logits.name

            # self.logits = logits * tf.tile(self.prefactor_tensor,
            #                                 tf.shape(self.logits)[0])
            self.logits = logits*self.prefactor
        return self.logits, None

    def get_n_output(self):
        return self.targets
