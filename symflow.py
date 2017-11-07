import numpy as np
import tensorflow as tf
# import preprocessing
# from preprocessing import Dataset, RadParameters, AngParameters
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import ml_util as ml
import pickle


class Network():

    def __init__(self, nodes):
        self.nodes = nodes
        self.rand_state = np.random.get_state()
        self.graph = None
        self.target_mean = 0
        self.target_std = 1
        self.sess = None
        self.graph = None
        self.initialized = False
        self.optimizer = None
        self.checkpoint_path = None

        self.findall_nodes()

    def findall_nodes(self):

        nodes = []
        for n in self.nodes:
            nodes += n.findall_forward()
        nodes = dict((k,1) for k in nodes)
        self.nodes = list(nodes.keys())

    def train(self,
              step_size=0.01,
              max_steps=50001,
              b_=0,
              verbose=True,
              optimizer=None,
              adaptive_rate=False):

        """ Train the master neural network

        Parameters:
        ----------
        step_size: float; step size for gradient descent
        max_steps: int; number of training epochs
        b: float; regularization parameter
        verbose: boolean; print cost for intermediate training epochs
        optimizer: {tf.nn.GradientDescentOptimizer,tf.nn.AdamOptimizer, ...}
        adaptive_rate: boolean; wether to adjust step_size if cost increases
                        not recommended for AdamOptimizer

        Returns:
        --------
        None
        """


        self.model_loaded = True
        if self.graph is None:
            self.graph = tf.Graph()

        with self.graph.as_default():

            if self.sess == None:
                sess = tf.Session()
                self.sess = sess
            else:
                sess = self.sess


            # Build all the required tensors
            logits_list = self.build_network()
            cost_list = self.get_cost(logits_list)
            train_feed_dict, valid_feed_dict = self.get_feed('train')

            cost = 0
            for c in cost_list:
                cost += c

            # # L2-loss
            # loss = 0
            # with tf.variable_scope("", reuse=True):
            #     for net in self.subnets:
            #         if isinstance(net,list):
            #             for net in net:
            #                 for l, layer in enumerate(net.layers):
            #                     name = net.species
            #                     loss += tf.nn.l2_loss(tf.get_variable("{}/W{}".format(name, l+1))) * \
            #                             b[name]/layer
            #         else:
            #             for l, layer in enumerate(net.layers):
            #                 name = net.species
            #                 loss += tf.nn.l2_loss(tf.get_variable("{}/W{}".format(name, l+1))) * \
            #                     b[name]/layer
            #
            # cost += loss          adaptive_rate=False
            #
            # for i, s in enumerate(species):
            #     train_feed_dict['{}/b:0'.format(s)] = b_[i]
            #     valid_feed_dict['{}/b:0'.format(s)] = 0


            if self.optimizer == None:
                if optimizer == None:
                    self.optimizer = tf.train.AdamOptimizer(learning_rate = step_size)
                else:
                    self.optimizer = optimizer

            train_step = self.optimizer.minimize(cost)

            # Workaround to load the AdamOptimizer variables
            if not self.checkpoint_path == None:
                saver = tf.train.Saver()
                saver.restore(self.sess,self.checkpoint_path)
                self.checkpoint_path = None

            ml.initialize_uninitialized(self.sess)

            self.initialized = True

            # train_writer = tf.summary.FileWriter('./          adaptive_rate=Falselog/',
            #                           self.graph)

            old_cost = 1e8

            for _ in range(0,max_steps):

                sess.run(train_step,feed_dict=train_feed_dict)

                # if _%int(max_steps/100) == 0 and adaptive_rate == True:
                #     new_cost = sess.run(tf.sqrt(cost),
                #         feed_dict=train_feed_dict)
                #
                #     if new_cost > old_cost:
                #         step_size /= 2
                #         print('Step size decreased to {}'.format(step_size))
                #         train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cost)
                #     old_cost = new_cost

                if _%int(max_steps/10) == 0 and verbose:
                    print('Step: ' + str(_))
                    print('Training set cost:')
                    if len(cost_list) > 1:
                        for i, c in enumerate(cost_list):
                            print('{}: {}'.format(i,sess.run(tf.sqrt(c),feed_dict=train_feed_dict)))
                    print('Total: {}'.format(sess.run(tf.sqrt(cost),feed_dict=train_feed_dict)))
                    print('Validation set cost:')
                    if len(cost_list) > 1:
                        for i, c in enumerate(cost_list):
                            print('{}: {}'.format(i,sess.run(tf.sqrt(c),feed_dict=valid_feed_dict)))
                    print('Total: {}'.format(sess.run(tf.sqrt(cost),feed_dict=valid_feed_dict)))
                    print('--------------------')
                    # print('L2-loss: {}'.format(sess.run(loss,feed_dict=train_feed_dict)))

    def get_logits(self, summarize = True, which = 'train'):
        """ Uses trained model on training or test sets

        Parameters:
        -----------
        summarize: boolean; Should the subnet structure be used to sum ouputs
                    accordingly?
        which: {'train','test'}; which set logits are computed for

        Returns:
        --------
        [numpy.array]; resulting logits grouped by independent subnet datasets
        """


        with self.graph.as_default():

            logits_list = self.build_network()

            sess = self.sess
            feed_dict, _ = self.get_feed(train_valid_split = 1.0, which = which)

            return_list = []

            for logits in logits_list:
                return_list.append(sess.run(logits,feed_dict=feed_dict))

            return return_list

    def set_nodes(self, nodes):
        if isinstance(nodes,list):
            self.nodes = nodes
        else:
            raise Exception("Error: list of nodes expected")

    def find_datanodes(self):
        """ Find the datanodes of in the network
        """
        roots = []

        for n in self.nodes:
            if isinstance(n, Datanode):
                roots.append(n)

        return roots

    def find_targetnodes(self):

        self.connect_backwards()

        targetnodes = []
        for n in self.find_datanodes():
            if len(n.receives_from) > 0:
                targetnodes.append(n)
        return targetnodes


    def connect_backwards(self):
        """ Automatically determine receives_from attribute of subnet nodes
        """

        for n in self.nodes:
            n.receives_from = []

        for n1 in self.nodes:
            for n2 in n1.sends_to:
                n2.receives_from.append(n1)

    def cost_function(self, y1, y2):
        return tf.reduce_mean(tf.reduce_mean(tf.square(y1-y2),0))

    def build_network(self):
        """ Builds the network by starting at the Subnetnodes connected
        to the ouput Datanodes and connecting nodes recursively backwards.
        """

        logits_list = []
        for dn in self.find_datanodes():

            if len(dn.receives_from) == 0: continue

            logits = 0
            for rf in dn.receives_from:
                logits += rf.get_tensors(rf.connect_backwards())[0]

            logits_list.append(logits)

        return logits_list

    def get_cost(self, logits_list):

        cost_list = []

        for dn, logits in zip(self.find_targetnodes(), logits_list):

            if len(dn.receives_from) == 0: continue
            _, y_ = dn.get_tensors()
            cost_list.append(self.cost_function(y_, logits))

        return cost_list

    def get_feed(self, which = 'train', train_valid_split = 0.8, seed = 42):
        """ Return a dictionary that can be used as a feed_dict in tensorflow

        Parameters:
        -----------
        which: {'train',test'}, which part of the dataset is used
        train_valid_split: float; ratio of train and validation set size
        seed: int; seed parameter for the random shuffle algorithm, make

        Returns:
        --------
        (dictionary, dictionary): either (training feed dictionary, validation feed dict.)
                                or (testing feed dictionary, None)
        """
        train_feed_dict = {}
        valid_feed_dict = {}
        test_feed_dict = {}

        for dn in self.find_datanodes():
            train_feed_dict.update(dn.get_feed('train', train_valid_split, seed))
            valid_feed_dict.update(dn.get_feed('valid', train_valid_split, seed))
            test_feed_dict.update(dn.get_feed('test', seed = seed))

        if which == 'train':
            return train_feed_dict, valid_feed_dict
        elif which == 'test':
            return test_feed_dict, None



class Node():

    def __init__(self):
        self.sends_to = []
        self.receives_from = []

    def __rshift__(self, other):
        self.update_sends_to(other)
        return other

    def __lshift__(self, other):
        other.update_sends_to(self)
        return other

    def update_sends_to(self, to):

        contained = False
        for s in self.sends_to:
            if s == to:
                contained = True
        if not contained:
            self.sends_to.append(to)

    def sends_to_names(self):
        names = []
        for to in self.sends_to:
            names.append(to.name)
        return names

    def findall_forward(self):
        nodes = [self]
        for n in self.sends_to:
            if isinstance(self, Subnetnode) and isinstance(n,Datanode):
                continue
            nodes += n.findall_forward()
        return nodes

class Datanode(Node):

    seed = 42
    index = 0

    def __init__(self, X = None, y = None, scaler = None):

        self.scaler = scaler
        self.name = 'Datanode'

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.x_name = None
        self.y_name = None
        self.x = None
        self.y_ = None
        super().__init__()

        # Keep track of Datanode instances and index them
        self.index = Datanode.index
        Datanode.index += 1

        if isinstance(X,np.ndarray) and isinstance(y, np.ndarray):
            self.add_dataset(X,y)
        elif isinstance(X,np.ndarray):
            self.add_dataset(X)


    def add_dataset(self, X, y = 0, test_size = 0.2, scale = True):

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
                print('No scaler provided, using MinMaxScaler fitted to ' +
                'training set')
                self.scaler = scaler
            else:
                scaler = self.scaler

            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        if not isinstance(self.X_train, np.ndarray):
            print('No dataset contained yet, initializing node with this dataset.')
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

class Subnetnode(Node):

    seed = 42

    def __init__(self, name, targets):
        self.name = name
        self.constructor = ml.fc_nn
        self.logits_name = None
        self.layers = [32] * 3
        self.targets = targets
        self.activations = [tf.nn.sigmoid] * 3
        self.features = 0
        self.logits = None
        super().__init__()

    def get_n_output(self):
        return self.targets

    def get_tensors(self, x):
        """ Builds the subnetwork by defining logits

        Parameters:
        -----------
        x: tf.placeholder for input data
        train_feed_dict.update(subnet.get_feed('train', train_valid_split, seed))
                valid_feed_dict.update(subnet.get_feed('valid', train_valid_split, seed))
                test_feed_dict.update(subnet.get_feed('test', seed = seed))
        Returns:
        ---------
        logits: tf.placeholder for output data

        """
        if self.logits_name == None:

            with tf.variable_scope(self.name) as scope:
                try:
                    logits = self.constructor(self, x)
                except ValueError:
                    print('Sharing variables')
                    scope.reuse_variables()
                    logits = self.constructor(self, x)

            self.logits_name = logits.name
            self.logits = logits

        return self.logits, None

    def connect_backwards(self):

        input_tensors = []

        #Automatically determine size of input tensor for this subnet
        n_features = 0

        for rf in self.receives_from:
            print('Connecting {} to {}'.format(self.name, rf.name))
            input_tensors.append(rf.get_tensors(rf.connect_backwards())[0])
            n_features += rf.get_n_output()

        self.features = n_features
        if len(input_tensors) > 1:
            return tf.concat(input_tensors, axis = 1)
        else:
            return input_tensors[0]
