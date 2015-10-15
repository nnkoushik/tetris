"""
 
 We a least p-norm fit layer for linear regression


 References :
   - http://deeplearning.net/tutorial/logreg.html
"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

class ValueFunction(object):
    def __init__(self, input, n_in, p = 2):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type p: int
        :param p: The kind of norm we will use
        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.random.rand(
                n_in, 1).astype(
                theano.config.floatX)
            ,
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.ones(
                (1,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def compute_val(self, inp_vec):
        return T.dot(inp_vec, self.W) + self.b

    def cost(self, y):
        #print(self.input.get_value().shape)
        print(self.W.get_value().shape)
        c = T.dot(self.input, self.W) + self.b - y
        #print(c.get_value().shape)
        return T.mean(abs(c))
