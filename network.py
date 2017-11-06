""" Module that implements Network, a class that combines several subnets
to build a master neural network that can be trained on datasets.
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
import ml_util as ml
from sklearn.model_selection import train_test_split

import subnet
from ml_util import *
import preprocessing as pre
from matplotlib import pyplot as plt
import math
import pickle
from collections import namedtuple
Dataset = namedtuple("Dataset", "data species species_index n")

class Network():

    def __init__(self, subnets):

        if not isinstance(subnets, list):
            self.subnets = [subnets]
        else:
            self.subnets = subnets

        self.model_loaded = False
        self.rand_state = np.random.get_state()
        self.graph = None
        self.target_mean = 0
        self.target_std = 1
        self.sess = None
        self.graph = None
        self.initialized = False
        self.optimizer = None
        self.checkpoint_path = None

    # ========= Network operations ============ #

    def __add__(self, other):
        if isinstance(other, subnet.Subnet):
            if not len(self.subnets) == 1:
                raise Exception(" + operator only valid if only one training set contained")
            else:
                self.subnets[0] += [other]
        else:
            raise Exception("Datatypes not compatible")

        return self

    def __and__(self, other):
        if isinstance(other, subnet.Subnet):
            self.subnets += [[other]]
        elif isinstance(other, Network):
            self.subnets += other.subnets
        else:
            raise Exception("Datatypes not compatible")

        return self

    def reset(self):
        self.sess = None
        self.graph = None
        self.initialized = False
        self.optimizer = None
        self.checkpoint_path = None

    def construct_network(self):
        """ Builds the tensorflow graph from subnets
        """

        cnt = 0
        logits = []
        for subnet in self.subnets:
            if isinstance(subnet,list):
                sublist = []
                for s in subnet:
                    sublist.append(s.get_logits(cnt)[0])
                    cnt += 1
                logits.append(sublist)
            else:
                logits.append(subnet.get_logits(cnt)[0])
                cnt += 1

        return logits

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

        for subnet in self.subnets:
            if isinstance(subnet,list):
                for s in subnet:
                    train_feed_dict.update(s.get_feed('train', train_valid_split, seed))
                    valid_feed_dict.update(s.get_feed('valid', train_valid_split, seed))
                    test_feed_dict.update(s.get_feed('test', train_valid_split, seed))
            else:
                train_feed_dict.update(subnet.get_feed('train', train_valid_split, seed))
                valid_feed_dict.update(subnet.get_feed('valid', train_valid_split, seed))
                test_feed_dict.update(subnet.get_feed('test', seed = seed))

        if which == 'train':
            return train_feed_dict, valid_feed_dict
        elif which == 'test':
            return test_feed_dict, None


    def get_cost(self):
        """ Build the tensorflow node that defines the cost function

        Returns
        -------
        cost_list: [tensorflow.placeholder]; list of costs for subnets. subnets
            whose outputs are added together share cost functions
        """
        cost_list = []

        for subnet in self.subnets:
            if isinstance(subnet,list):
                cost = 0
                y_ = self.graph.get_tensor_by_name(subnet[0].y_name)
                log = 0
                for s in subnet:
                    log += self.graph.get_tensor_by_name(s.logits_name)
                cost += tf.reduce_mean(tf.reduce_mean(tf.square(y_-log),0))
            else:
                log = self.graph.get_tensor_by_name(subnet.logits_name)
                y_ = self.graph.get_tensor_by_name(subnet.y_name)
                cost = tf.reduce_mean(tf.reduce_mean(tf.square(y_-log),0))
            cost_list.append(cost)

        return cost_list




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
            build_graph = True
        else:
            build_graph = False

        with self.graph.as_default():

            if self.sess == None:
                sess = tf.Session()
                self.sess = sess
            else:
                sess = self.sess



            # Get number of distinct subnet species
            species = {}
            for net in self.subnets:
                if isinstance(net,list):
                    for net in net:
                        for l,_ in enumerate(net.layers):
                            name = net.species
                            species[name] = 1
                else:
                    for l,_ in enumerate(net.layers):
                        name = net.species
                        species[name] = 1
            n_species = len(species)


            # Build all the required tensors
            b = {}
            if build_graph:
                self.construct_network()
                for s in species:
                    b[s] = tf.placeholder(tf.float32,name = '{}/b'.format(s))
            else:
                for s in species:
                    b[s] = self.graph.get_tensor_by_name('{}/b:0'.format(s))

            cost_list = self.get_cost()
            train_feed_dict, valid_feed_dict = self.get_feed('train')
            cost = 0
            for c in cost_list:
                cost += c

            # L2-loss
            loss = 0
            with tf.variable_scope("", reuse=True):
                for net in self.subnets:
                    if isinstance(net,list):
                        for net in net:
                            for l, layer in enumerate(net.layers):
                                name = net.species
                                loss += tf.nn.l2_loss(tf.get_variable("{}/W{}".format(name, l+1))) * \
                                        b[name]/layer
                    else:
                        for l, layer in enumerate(net.layers):
                            name = net.species
                            loss += tf.nn.l2_loss(tf.get_variable("{}/W{}".format(name, l+1))) * \
                                b[name]/layer

            cost += loss

            for i, s in enumerate(species):
                train_feed_dict['{}/b:0'.format(s)] = b_[i]
                valid_feed_dict['{}/b:0'.format(s)] = 0


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

            for _ in range(0,max_steps):

                sess.run(train_step,feed_dict=train_feed_dict)

                if _%int(max_steps/100) == 0 and adaptive_rate == True:
                    new_cost = sess.run(tf.sqrt(cost),
                        feed_dict=train_feed_dict)

                    if new_cost > old_cost:
                        step_size /= 2
                        print('Step size decreased to {}'.format(step_size))
                        train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cost)
                    old_cost = new_cost

                if _%int(max_steps/10) == 0 and verbose:
                    print('Step: ' + str(_))
                    print('Training set loss:')
                    if len(cost_list) > 1:
                        for i, c in enumerate(cost_list):
                            print('{}: {}'.format(i,sess.run(tf.sqrt(c),feed_dict=train_feed_dict)))
                    print('Total: {}'.format(sess.run(tf.sqrt(cost-loss),feed_dict=train_feed_dict)))
                    print('Validation set loss:')
                    if len(cost_list) > 1:
                        for i, c in enumerate(cost_list):
                            print('{}: {}'.format(i,sess.run(tf.sqrt(c),feed_dict=valid_feed_dict)))
                    print('Total: {}'.format(sess.run(tf.sqrt(cost),feed_dict=valid_feed_dict)))
                    print('--------------------')
                    print('L2-loss: {}'.format(sess.run(loss,feed_dict=train_feed_dict)))

    def predict(self, data, species, species_index, processed = True, verbose = False):

        if data.ndim == 2:
            data = data.reshape(-1,1,data.shape[1])
        else:
            raise Exception('data.ndim != 2')

        ds = Dataset(data, species, species_index,1)
        targets = np.zeros(data.shape[0])
        snet = subnet.Subnet()

        found = False
        for s in self.subnets:
            if found == True:
                break
            if isinstance(s,list):
                for s2 in s:
                    if s2.species_index == ds.species_index:
                        snet.scaler = s2.scaler
                        if verbose: print("Sharing scaler with species " + s2.species)
                        found = True
                        break
            else:
                if s.species_index == ds.species_index:
                    snet.scaler = s.scaler
                    if verbose: print("Sharing scaler with species " + s.species)
                    break

        snet.add_datasets([ds], targets, processed = processed, test_size=0.0)
        self = self % snet

        result = self.get_logits()[-1]

        del self.subnets[-1]

        return result

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

        if not self.model_loaded:
            raise Exception('Model not loaded!')
        else:
            with self.graph.as_default():

                logits_list = self.construct_network()

                sess = self.sess
                feed_dict, _ = self.get_feed(train_valid_split = 1.0, which = which)

                return_list = []

                for logits in logits_list:
                    if isinstance(logits,list):
                        result = 0
                        for logits in logits:
                            if summarize:
                                result += sess.run(logits,feed_dict=feed_dict)
                            else:
                                return_list.append(sess.run(logits,feed_dict=feed_dict))
                        if summarize:
                            return_list.append(result)
                    else:
                        return_list.append(sess.run(logits,feed_dict=feed_dict))
                return return_list

    def save_model(self, path):
        """ Save trained model
        """

        if path[-5:] == '.ckpt':
            path = path[:-5]

        with self.graph.as_default():
            sess = self.sess
            saver = tf.train.Saver()
            saver.save(sess,save_path = path + '.ckpt')

    def restore_model(self, path):
        """ Load trained model from path
        """

        if path[-5:] == '.ckpt':
            path = path[:-5]

        self.checkpoint_path = path + '.ckpt'
        g = tf.Graph()
        with g.as_default():
            sess = tf.Session()
            self.construct_network()
            b = tf.placeholder(tf.float32,name = 'b')
            saver = tf.train.Saver()
            saver.restore(sess,path + '.ckpt')
            self.model_loaded = True
            self.sess = sess
            self.graph = g
            self.initialized = True

    def save_all(self, net_dir, override = False):
        """ Saves the model including all subnets and datasets
        using pickle
        """
        try:
            os.mkdir(net_dir)
        except FileExistsError:
            if override:
                print('Overriding...')
                import shutil
                shutil.rmtree(net_dir)
                os.mkdir(net_dir)
            else:
                print('Directory/Network already exists. Network not saved...')
                return None

        # Pickle does not seem to be compatible with tensorflow so just
        # save subnetworks with it
        with open(os.path.join(net_dir,'subnets'),'wb') as file:
            pickle.dump(self.subnets,file)

        self.save_model(os.path.join(net_dir,'model'))



    def load_all(self, net_dir):
        """ Loads the model including all subnets and datasets
        using pickle
        """

        with open(os.path.join(net_dir,'subnets'),'rb') as file:
            self.subnets = pickle.load(file)

        self.restore_model(os.path.join(net_dir,'model'))

def load_network(path):
    network = Network([])
    network.load_all(path)
    return network
