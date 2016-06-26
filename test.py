#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from epsfree.mdn import MDN


batch_size = 64
nsamp = batch_size * 5000
nin = 2
nhidden = 2
ngauss = 2
nout = 1
nu = (nout * (nout + 1)) // 2
nvar = 512

x_data = np.random.rand(nsamp, nin)
theta_data = np.atleast_2d(np.dot(x_data, np.array([0.5, -0.1]))).T

model = MDN(nin, nout, nhidden, ngauss, nvar)
