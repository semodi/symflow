import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from . import ml_util as ml
import pickle

class Graph():

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
