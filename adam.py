# -*- coding: utf-8 -*-

"""
http://arxiv.org/pdf/1412.6980v8.pdf

"""

from __future__ import division, print_function

import theano
import theano.tensor as T

__all__ = ["get_adam_updates"]

def get_adam_updates(cost, params,
                     alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
    grads = T.grad(cost, params)
    t = theano.shared(1.0)
    updates = [(t, t + 1.0)]
    alpha_t = alpha * T.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)

    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.0)
        v = theano.shared(p.get_value() * 0.0)

        m_t = beta1 * m + (1.0 - beta1) * g
        v_t = beta2 * v + (1.0 - beta2) * g**2
        p_t = p - alpha_t * m_t / (T.sqrt(v_t) + eps)

        updates += [(p, p_t), (m, m_t), (v, v_t)]

    return updates
