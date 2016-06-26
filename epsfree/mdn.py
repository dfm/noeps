# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .adam import get_adam_updates

__all__ = ["MDN"]


class MDN(object):

    def __init__(self, nin, nout, nhidden, ngauss, nvar):
        self.nin = nin
        self.nout = nout
        self.nhidden = nhidden
        self.ngauss = ngauss
        self.nu = nu = (self.nout * (self.nout + 1)) // 2

        # In/out:
        x = T.dmatrix("x")
        theta = T.dmatrix("theta")

        # Parameters:
        W = theano.shared(np.random.rand(nin, nhidden), name="W")
        b = theano.shared(np.random.rand(nhidden), name="b")
        y = T.dot(x, W) + b

        W_alpha = theano.shared(1e-8*np.random.rand(nhidden, ngauss),
                                name="W_alpha")
        b_alpha = theano.shared(np.zeros(ngauss), name="b_alpha")
        alpha = T.nnet.softmax(T.dot(y, W_alpha) + b_alpha)

        W_mk = theano.shared(1e-8*np.random.rand(nhidden, ngauss*nout),
                             name="W_mk")
        b_mk = theano.shared(np.zeros((ngauss*nout)), name="b_mk")

        W_u = theano.shared(1e-8*np.random.rand(nhidden, ngauss*nu),
                            name="W_u")
        b_u = theano.shared(np.zeros((ngauss*nu)), name="b_u")

        # Compute the Gaussian cost using a reduce:
        Uvals = T.dot(y, W_u) + b_u
        mkvals = T.dot(y, W_mk) + b_mk

        def apply_gaussian(Uv, mk, a, th, current):
            for i in range(ngauss):
                arg = T.exp(Uv[i*nu:i*nu+nout])
                current += T.sum(arg)
                U = T.diag(arg)
                U = T.set_subtensor(U[np.triu_indices(nout, 1)],
                                    Uv[i*nu+nout:(i+1)*nu])
                r = th - mk[i*nout:(i+1)*nout]
                r2 = T.dot(r, T.dot(U.T, T.dot(U, r)))
                current += T.log(a[i]) - 0.5 * r2
            return current

        outputs_info = T.as_tensor_variable(np.asarray(0.0, float))
        lnprob, _ = theano.reduce(apply_gaussian,
                                  [Uvals, mkvals, alpha, theta],
                                  outputs_info)
        cost = -lnprob

        self.params = [W, b, W_alpha, b_alpha, W_mk, b_mk, W_u, b_u]
        self.grads = T.grad(cost, self.params)
        updates = get_adam_updates(cost, self.params)

        self.update_step = theano.function([x, theta], outputs=cost,
                                           updates=updates)
        self.cost_func = theano.function([x, theta], outputs=cost)

        # Stochastic objective:
        ntot = np.sum([np.prod(np.shape(p.get_value())) for p in self.params])
        rng = RandomStreams()
        u = rng.normal((nvar, ntot))
        phi_m = theano.shared(np.zeros(ntot), name="phi_m")
        phi_s = theano.shared(np.zeros(ntot), name="phi_s")
        phi = (phi_m + T.exp(0.5 * phi_s))[None, :] * u

        print(theano.function([], outputs=phi)().shape)

    def training_update(self, x, theta):
        return self.update_step(x, theta)

    def compute_cost(self, x, theta):
        return self.cost_func(x, theta)
