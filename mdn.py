#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import time
import numpy as np

import theano
import theano.tensor as T

from adam import get_adam_updates

__all__ = []

batch_size = 64
batch_offset = T.lscalar()

nsamp = batch_size * 5000
nin = 2
nhidden = 2
ngauss = 2
nout = 1
nu = (nout * (nout + 1)) // 2
print(nu, nout)


x_data = np.random.rand(nsamp, nin)
# theta_data = np.atleast_2d(np.dot(x_data, np.array([0.5, -0.1]))).T
theta_data = np.random.randn(nsamp, nout)
# theta_data += 1e-3*np.random.randn(*(theta_data.shape))

# In/out:
x = T.dmatrix("x")
theta = T.dmatrix("theta")

# Parameters:
W = theano.shared(np.random.rand(nin, nhidden), name="W")
b = theano.shared(np.random.rand(nhidden), name="b")
y = T.dot(x, W) + b

W_alpha = theano.shared(1e-8*np.random.rand(nhidden, ngauss), name="W_alpha")
b_alpha = theano.shared(np.zeros(ngauss), name="b_alpha")
alpha = T.nnet.softmax(T.dot(y, W_alpha) + b_alpha)

W_mk = theano.shared(1e-8*np.random.rand(nhidden, ngauss*nout), name="W_mk")
b_mk = theano.shared(np.zeros((ngauss*nout)), name="b_mk")

W_u = theano.shared(1e-8*np.random.rand(nhidden, ngauss*nu), name="W_u")
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
lnprob, _ = theano.reduce(apply_gaussian, [Uvals, mkvals, alpha, theta],
                          outputs_info)
cost = -lnprob
params = [W, b, W_alpha, b_alpha, W_mk, b_mk, W_u, b_u]

for p in params:
    cost += 0.1 * T.sum(p**2)

grads = T.grad(cost, params)

updates = get_adam_updates(cost, params)

strt = time.time()
update_step = theano.function([x, theta], outputs=cost, updates=updates)
cost_func = theano.function([x, theta], outputs=cost)
print(time.time() - strt)

# strt = time.time()
# func(x_data, theta_data)
# print(time.time() - strt)

vals = []
ind = 0
for i in range(1000):
    if (i + 1) % 50 == 0:
        print(i, 0.9**i, cost_func(x_data, theta_data))
    inds = np.random.randint(nsamp, size=batch_size)
    vals.append(update_step(x_data    [inds],
                            theta_data[inds]))
    ind = (ind + batch_size) % len(x_data)

import matplotlib.pyplot as pl
pl.plot(vals)
pl.savefig("vals.png")
