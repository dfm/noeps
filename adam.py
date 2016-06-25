# -*- coding: utf-8 -*-

from __future__ import division, print_function

import theano
import theano.tensor as T

__all__ = ["get_adam_updates"]

def get_adam_updates(cost, params,
                     alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
    grads = T.grad(cost, params)
    t = theano.shared(1.0)
    updates = [(t, t + 1.0)]

    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.0)
        v = theano.shared(p.get_value() * 0.0)

        m_t = beta1 * m + (1.0 - beta1) * g
        m_hat = m_t / (1.0 - beta1**t)
        v_t = beta2 * v + (1.0 - beta2) * g**2
        v_hat = v_t / (1.0 - beta2**t)
        p_t = p - alpha * m_hat / (T.sqrt(v_hat) + eps)

        updates += [(p, p_t), (m, m_t), (v, v_t)]

    return updates
