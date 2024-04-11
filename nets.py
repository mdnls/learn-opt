import jax
from jax import numpy as jnp, random, tree_util
from utils import regr_to_steps, mtx_optimal_regr
class ReLU_MLP():
    def __init__(self, dims, seed):
        '''
        Construct a parameter tree for MLP with bias. intermediate activations are ReLU and final activation is affine
        :param dims: a list of integers of the form [input_dim, output_dim, ..., output_dim]. The resulting MLP has k-1
            layers where k=len(dims)
        :param seed: input random seed
        :return: parameter tree for MLP, list of seeds of [k-1 x 2]
        '''
        params = []
        keys = random.split(seed, len(dims)-1)

        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:])):
            params.append({
                f"W": random.normal(keys[i], (d_in, d_out)),
                f"b": random.normal(keys[i], (d_out,))
            })

        self.params = params
        self.keys = keys
    @staticmethod
    def fwd(params, x):
        input = x
        for layer in params[:-1]:
            output = input @ layer['W'] + layer['b']
            input = jnp.maximum(0, output)

        final_output = input @ params[-1]['W'] + params[-1]['b']
        return final_output

class GD():
    def __init__(self, step_sizes, gradient):
        self.params = {
            "grad": gradient,
            "step_sizes": step_sizes
        }

    @staticmethod
    def fwd(params, x):
        input = x
        grad = params['grad']
        for eta in params['step_sizes']:
            input = input - eta * grad(input)
        return input

    @staticmethod
    def constant_steps(A, n_steps):
        evals, _ = jnp.linalg.eigh(A)
        return (2/jnp.max(evals)) * jnp.ones((n_steps))

    @staticmethod
    def vieta_steps(A, n_steps):
        return regr_to_steps(mtx_optimal_regr(A, n_steps))

