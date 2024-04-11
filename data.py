import jax
from jax import scipy as jsp, numpy as jnp, random, tree_util


class Potential():
    def sample(self, sample_size, seed):
        pass
    @staticmethod
    def push(params, x):
        ''' Given x, return the nearest minimizer of potential'''
        return x

    @staticmethod
    def potential(params, x):
        ''' compute the potential at x'''
        return 0

    @staticmethod
    def grad(params, x):
        ''' compute gradient of the potential at x'''
        return 0

    @staticmethod
    def hessian(params, x):
        return 0

class VoronoiP(Potential):
    def __init__(self, modes, A=None):
        '''
        This potential function is V(x) = 1/2 min_(i=1...n) (x - xi) @ A @ (x - xi)
        :param modes: a matrix of size N x D with N vectors of D dimensions
        :param A: a square positive definite matrix. If None, A=Identity
        '''
        self.params = {
            "modes": modes,
            "A": A if (A is not None) else jnp.eye(modes.shape[1])
        }
    @staticmethod
    def push(params, x):
        deltas = params['modes'] - x.reshape((1, -1))
        dists = jnp.sum((deltas @ params['A']) * deltas, axis=1)
        return params['modes'][jnp.argmin(dists)]

    @staticmethod
    def potential(params, x):
        deltas = params['modes'] - x.reshape((1, -1))
        dists = jnp.sum((deltas @ params['A']) * deltas, axis=1)
        return 0.5 * jnp.min(dists)

    @staticmethod
    def grad(params, x):
        deltas = params['modes'] - x.reshape((1, -1))
        dists = deltas @ params['A'] @ deltas
        return params['A'] @ (x - params[jnp.argmin(dists)])

    @staticmethod
    def hessian(params, x):
        return params['A']

class GaussianMixtureP(Potential):
    def __init__(self, mixture_means, temp, prec=None):
        '''
        This potential function is V(x) = -(1/temp) log sum exp( -temp/2 * (x-xi) prec (x-xi) )
        :param mixture_means: a matrix of size N x D with N vectors of D dimensions
        :param temp: a scalar temperature parameter
        :param prec: the precision of the gaussian modes (D x D)
        '''

        # use 10 steps of newton method to approximate true maximizers

        self.params = {
            "means": mixture_means,
            "temp": temp,
            "A": prec if (prec is not None) else jnp.eye(mixture_means.shape[1])
        }

        mixture_modes = []

        for mean in mixture_means:
            itr = mean
            for i in range(10):
                hess = jax.hessian(lambda x: GaussianMixtureP.potential(self.params, x))(itr)
                grad = GaussianMixtureP.grad(self.params, itr)
                itr = itr - jnp.linalg.solve(hess, grad)
            mixture_modes.append(itr)

        self.params['modes'] = jnp.asarray(mixture_modes)

    @staticmethod
    def push(params, x):
        deltas = params['modes'] - x.reshape((1, -1))
        dists = jnp.sum((deltas @ params['A']) * deltas, axis=1)
        return params['modes'][jnp.argmin(dists)]

    @staticmethod
    def potential(params, x):
        deltas = params['means'] - x.reshape((1, -1))
        return (-1/params['temp']) * jnp.log(jnp.sum(jnp.exp( - (params['temp']/2) * jnp.sum((deltas @ params['A']) * deltas, axis=1))))

    @staticmethod
    def grad(params, x):
        return jax.grad(lambda x: GaussianMixtureP.potential(params, x))(x)

    @staticmethod
    def hessian(params, x):
        return jax.hessian(lambda x: GaussianMixtureP.potential(params, x))(x)


class Data():
    @staticmethod
    def sample(params, seed, N_samples):
        pass


class GaussianData():
    def __init__(self, mean, cov):
        self.params = {
            "mean": mean,
            "sqcov": jnp.real(jsp.linalg.sqrtm(cov)),
            "d": len(mean)
        }

    @staticmethod
    def sample(params, seed, N_samples):
        return random.normal(seed, (N_samples, params['d'])) @ params['sqcov'] + params['mean']

    @staticmethod
    def batch(params, potential, seed, N_samples):
        X = GaussianData.sample(params, seed, N_samples)
        Y = jax.vmap(lambda x: potential.push(potential.params, x))(X)
        return (X, Y)
