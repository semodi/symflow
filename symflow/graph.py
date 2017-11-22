import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from .nodes import *
import pandas as pd
from . import ml_util as ml
import pickle
from time import time

class Graph():

    def __init__(self, nodes):
        if isinstance(nodes, Datanode):
            nodes = [nodes]

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

# ================== Graph functions ======================

    def findall_nodes(self):
        """ Starting from Datanodes, do a forward recursive
        search to find all other nodeds in the graph
        """

        nodes = []
        for n in self.nodes:
            nodes += n.findall_forward()

        # Make sure list only contains every element once
        nodes = dict((k,1) for k in nodes)
        self.nodes = list(nodes.keys())
        self.connect_backwards()

    def set_nodes(self, nodes):
        if isinstance(nodes,list):
            self.nodes = nodes
        else:
            raise Exception("Error: list of nodes expected")

    def find_datanodes(self):
        """ Returns list of all Datanodes in network
        """
        roots = []

        for n in self.nodes:
            if isinstance(n, Datanode):
                roots.append(n)

        return roots

    def find_targetnodes(self):
        """ Returns list of all Datanodes
        that receive data from Subnetnodes and therefore provide target values
        """

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

    def distinct_namescopes(self):
        """ Find all subnetnodes with distinct namescope
        """

        namescopes = []

        for n in self.nodes:
            if isinstance(n, Subnetnode) and n.name not in namescopes:
                namescopes.append(n.name)
        return namescopes

# ================== ML functions ======================

    def reset(self):
        self.sess = None
        self.graph = None
        self.initialized = False
        self.optimizer = None
        self.checkpoint_path = None

    def get_prediction(self, datanode, which = 'train'):
        """ Uses trained model on training or test sets

        Parameters:
        -----------
        which: {'train','test'}; which set logits are computed for

        datanode: string, Datanode or integer; Determines which datanode returned
        predictions are  associated with by either providing the Datanode name
        (string), the Datanode itself, or the Datanode index (int)

        Returns:
        --------
        numpy.ndarray; resulting predictions
        """


        which_node = -1
        for i, n in enumerate(self.find_targetnodes())  :
            if isinstance(datanode, Datanode):
                if datanode == n:
                    which_node = i
            elif isinstance(datanode, str):
                if datanode == n.name:
                    which_node = i
            elif isinstance(datanode, int):
                if datanode == n.index:
                    which_node = i
            else:
                raise Exception("Input for datanode not understood")

        with self.graph.as_default():

            logits_list = self.build_network()

            sess = self.sess
            feed_dict, _ = self.get_feed(train_valid_split = 1.0, which = which)


            predictions = sess.run(logits_list[which_node],feed_dict=feed_dict)

            return predictions

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

    def cost_function(self, y1, y2):
        return tf.reduce_mean(tf.square(y1-y2), axis = 0)

    def get_cost(self, logits_list):
        """ Returns a list of all the cost tensors, given a list of logits_list
        (obtained by calling build_network())
        """

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
        (dictionary, dictionary): either (training feed dictionary,
            validation feed dict.) or (testing feed dictionary, None)
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

    def train(self,
              step_size=0.01,
              max_steps=50001,
              b_=0,
              verbose=True,
              optimizer=None,
              log_training = False,
              batch_size = 0):

        """ Train the master neural network

        Parameters:
        ----------
        step_size: float; step size for gradient descent
        max_steps: int; number of training epochs
        b: float; regularization parameter
        verbose: boolean; print cost for intermediate training epochs
        optimizer: {tf.nn.GradientDescentOptimizer,tf.nn.AdamOptimizer, ...}
        batch_size: int; size of batches, if 0 use entire training set (default:0)
        Returns:
        --------
        statistics: dict; training statistics
        """


        self.model_loaded = True
        if self.graph is None:
            self.graph = tf.Graph()
            build_graph = True
        else:
            build_graph = False

        with self.graph.as_default():

            if self.sess == None:
                sess = tf.Session()
                self.sess = sess
            else:
                sess = self.sess

            for n in self.nodes:
                n.set_prefactors()

            # Build all the required tensors
            logits_list = self.build_network()
            cost_list = self.get_cost(logits_list)
            train_feed_dict, valid_feed_dict = self.get_feed('train')



            cost = 0
            for c in cost_list:
                cost += c

            # Create regularization parameters for every distinct namescope
            b = {}
            if build_graph:
                for ns in self.distinct_namescopes():
                    b[ns] = tf.placeholder(tf.float32, name = '{}/b'.format(ns))
            else:
                for ns in self.distinct_namescopes():
                    b[ns] = self.graph.get_tensor_by_name('{}/b:0'.format(ns))

            # L2-loss
            loss = 0
            with tf.variable_scope("", reuse = True):
                for n in self.nodes:
                    if not isinstance(n, Subnetnode): continue

                    for l, layer in enumerate(n.layers):
                        name = n.name
                        loss += tf.nn.l2_loss(tf.get_variable("{}/W{}".format(name, l+1))) * \
                            b[name]/layer

            cost += loss

            if b_ == 0:
                b_ = [0] * len(self.distinct_namescopes())

            for i, ns in enumerate(self.distinct_namescopes()):
                train_feed_dict['{}/b:0'.format(ns)] = b_[i]
                valid_feed_dict['{}/b:0'.format(ns)] = 0


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

            train_writer = tf.summary.FileWriter('./log/',
                                      self.graph)

            old_cost = 1e8

            statistics = {}
            statistics['time_trained'] = time()
            statistics['total_cost'] = []
            statistics['loss'] = []
            statistics['partial_cost'] = {}
            for t in self.find_targetnodes():
                statistics['partial_cost'][t.name] = []

            for _ in range(0,max_steps):

                if batch_size > 0:
                    start = 0
                    while(start != -1):
                        batch_feed_dict, start = ml.get_batch_feed(train_feed_dict,
                                                            start, batch_size)
                        sess.run(train_step, feed_dict = batch_feed_dict)
                else:
                    sess.run(train_step, feed_dict=train_feed_dict)

                # if _%int(max_steps/100) == 0 and adaptive_rate == True:
                #     new_cost = sess.run(tf.sqrt(cost),
                #         feed_dict=train_feed_dict)
                #
                #     if new_cost > old_cost:
                #         step_size /= 2
                #         print('Step size decreased to {}'.format(step_size))
                #         train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cost)
                #     old_cost = new_cost

                # Log training process
                if _%int(max_steps/100) == 0 and log_training:
                    statistics['total_cost'].append(sess.run(tf.sqrt(cost),
                        feed_dict=valid_feed_dict))
                    statistics['loss'].append(sess.run(loss,
                        feed_dict=valid_feed_dict))
                    if len(cost_list) > 1:
                        for t, c in zip(self.find_targetnodes(), cost_list):
                            statistics['partial_cost'][t.name].append(sess.run(tf.sqrt(c),
                                feed_dict=valid_feed_dict))

                # Print training process
                if _%int(max_steps/10) == 0 and verbose:
                    print('Step: ' + str(_))
                    print('Training set loss:')
                    if len(cost_list) > 1:
                        for t, c in zip(self.find_targetnodes(), cost_list):
                            print('{}: {}'.format(t.name,sess.run(tf.sqrt(c),
                                feed_dict=train_feed_dict)))
                    print('Total: {}'.format(sess.run(tf.sqrt(cost-loss),
                        feed_dict=train_feed_dict)))
                    print('Validation set loss:')
                    if len(cost_list) > 1:
                        for t, c in zip(self.find_targetnodes(), cost_list):
                            print('{}: {}'.format(t.name, sess.run(tf.sqrt(c),
                                feed_dict=valid_feed_dict)))
                    print('Total: {}'.format(sess.run(tf.sqrt(cost),
                        feed_dict=valid_feed_dict)))
                    print('--------------------')
                    print('L2-loss: {}'.format(sess.run(loss,
                        feed_dict=train_feed_dict)))

            # Final log entry

            statistics['total_cost'].append(sess.run(tf.sqrt(cost),
                feed_dict=valid_feed_dict))
            statistics['loss'].append(sess.run(loss,
                feed_dict=valid_feed_dict))
            if len(cost_list) > 1:
                for t, c in zip(self.find_targetnodes(), cost_list):
                    statistics['partial_cost'][t.name].append(sess.run(tf.sqrt(c),
                        feed_dict=valid_feed_dict))
            statistics['time_trained'] = time() - statistics['time_trained']
            return statistics
