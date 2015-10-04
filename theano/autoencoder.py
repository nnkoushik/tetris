"""
 
 We implement a stacked denoising autoencoder here.


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
   - http://deeplearning.net/tutorial/mlp.html
   - http://deeplearning.net/tutorial/dA.html
"""
__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

class Autoencoder(object):
    def __init__(self, rng, input, n_in, n_mid, W=None, bhid=None,
                bvis=None, activation=theano.tensor.nnet.sigmoid):
    """
    We assume all connections btw adjacent layers are present
    Weight matrix W is of shape (n_in,n_mid)
    We are assuming Tied weights.
    and the bias vector bhid is of shape (n_mid,).
    The bias vector bvis is of shape (n_in,)

    NOTE : The nonlinearity used here is sigmoid

    Hidden unit activation is given by: sig(dot(input,W) + b)

    :type rng: numpy.random.RandomState
    :param rng: a random number generator used to initialize weights

    :type input: theano.tensor.dmatrix
    :param input: a symbolic tensor of shape (n_examples, n_in)

    :type n_in: int
    :param n_in: dimensionality of input same as dimension of output

    :type n_mid: int
    :param n_mid: number of hidden units

    :type activation: theano.Op or function
    :param activation: Non linearity to be applied in the hidden
           layer
    """
        self.input = input
        if W is None:
            W_values = numpy.asarray(
            rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_mid)),
            high=numpy.sqrt(6. / (n_in + n_mid)),
            size=(n_in, n_mid)
            ),
            dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if bhid is None:
            b_values = numpy.zeros((n_mid,), dtype=theano.config.floatX)
            bhid = theano.shared(value=b_values, name='bhid', borrow=True)

        if bvis is None:
            b_values = numpy.zeros((n_in,), dtype=theano.config.floatX)
            bvis = theano.shared(value=b_values, name='bvis', borrow=True)


        self.W = W
        self.b = bhid
        self.bN = bvis
        self.WN = self.W.T

        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

        def computeHiddenSignals(self, input):
        """ Computes activation in the hidden layer """
            return self.activation(T.dot(input, self.W) + self.b)

        def computeFinalSignal(self, hidden):
        """ Computes final signal """
            return self.activation(T.dot(hidden, self.Wn) + self.Wn)

        def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input


        def computeUpdates(self, corruption, learning_rate):
            cor_x = self.get_corrupted_input(self.x, corruption)
            hid = computeHiddenSignals(self, cor_x)
            z = computeFinalSignal(self, hid)
            #note : we sum over the size of a datapoint; if we are using
            #minibatches, L will be a vector, with one entry per
            #example in minibatch
            # note : L is now a vector, where each element is the
            L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

            #cross-entropy cost of the reconstruction of the
            #corresponding example of the minibatch. We need to
            #compute the average of all these to get the cost of
            #the minibatch
            cost = T.mean(L)

            # compute the gradients of the cost of the `dA` with respect
            # to its parameters
            gparams = T.grad(cost, self.params)
            # generate the list of updates
            updates = [
                (param, param - learning_rate * gparam)
                for param, gparam in zip(self.params, gparams)
            ]

            return (cost, updates)

