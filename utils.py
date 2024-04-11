import jax
from jax import numpy as jnp, random, tree_util

def steps_to_regr(steps):
    '''
    convert step sizes to regression coefficients
    '''
    return (- jnp.ones(len(steps) + 1)) ** (jnp.arange(len(steps) + 1)) * symm_poly(steps)


def mtx_optimal_regr(A, k):
    '''
    compute the optimal regression coefficients for a matrix A
    '''
    eigs = jnp.linalg.eigvalsh(A)
    powers = jnp.stack([eigs ** j for j in range(k + 1)], axis=0)
    mtx = jnp.linalg.inv(powers @ powers.T)
    return mtx[:, 0] / mtx[0, 0]

def regr_to_steps(regr):
    '''
    convert regression coefficients to step sizes
    '''
    return jnp.roots(regr)

def symm_poly(inp):
    '''
    Evaluate the elementary symmetric polynomials wrt inp

    :param i: integer
    :param inp: a list
    :param cumulative: if true, return a list of length i of all symmetric polynomials up to
    :return: elementwise evaluation of ei
    '''

    outp = jnp.ones((len(inp) + 1, len(inp) + 1))

    for k in range(1, len(inp) + 1):
        for n in range(1, k):
            outp[k, n] = outp[k - 1, n] + outp[k - 1, n - 1] * inp[k - 1]
        outp[k, k] = inp[k - 1] * outp[k - 1, k - 1]

    return outp[-1]