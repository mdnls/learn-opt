from nets import ReLU_MLP, GD
from data import VoronoiP, GaussianMixtureP, GaussianData
import optax
import jax
from jax import numpy as jnp, random, grad
class Runner():
    def __init__(self, config):
        self.config = config

    @staticmethod
    def optimize_model(config, seed, model, potential, dataset):
        opt_cfg = config['train']['optimization']

        if opt_cfg['optimizer'] == 'Adam':
            params = model.params
            opt = optax.adam(learning_rate=opt_cfg['lr'], b1=opt_cfg['beta1'])
            opt_state = opt.init(params)

            bs = config['train']['optimization']['bs']
            _new_batch = lambda seed: dataset.batch(dataset.params, potential, seed, bs)

            def evaluate_mse(params, X, Y):
                return 0.5 * jnp.mean(jnp.linalg.norm( jax.vmap(lambda x: model.fwd(params, x))(X) - Y, axis=1)**2)
            mse_and_grad = jax.value_and_grad(evaluate_mse)

            losses = []
            for i in range(opt_cfg['steps']):
                oldseed, seed = random.split(seed)
                X, Y = _new_batch(seed)

                loss, grads = mse_and_grad(params, X, Y)
                update, opt_state = opt.update(grads, opt_state)
                params = optax.apply_updates(params, update)
                delta = params[0]['W'] - optax.apply_updates(params, update)[0]['W']
                losses.append(loss)
                print(f'Step: {i}, loss: {loss}')

            # TODO: reimplement in a way that respects immutability ie. by initializing a new struct of type type(model)
            model.params = params
            return model, losses
        else:
            raise ValueError('This optimizer is not implemented')
    @staticmethod
    def init_model(config, seed):
        if config['model']['architecture'] == 'ReLU_MLP':
            model = ReLU_MLP(config['model']['dims'], seed)
            return model
        elif config['model']['architecture'] == 'Vieta_GD':
            potential = Runner.init_potential(config)
            # TODO: smarter way to get the hessian that is aware of non-constant hessians for potentially non-diff functions
            hess = potential.hessian(potential.params, jnp.zeros((config['dim'],)))
            steps = GD.vieta_steps(hess, config['model']['steps'])
            model = GD(steps, lambda x: potential.grad(potential.params, x))
            return model

    @staticmethod
    def init_potential(config):
        if config['train']['potential']['type'] == 'BimodalVoronoi':
            d = config['dim']
            modes = jnp.asarray([-jnp.ones((d,)), jnp.ones((d,))])
            return VoronoiP(modes)
        if config['train']['potential']['type'] == 'GaussianMixture':
            raise ValueError('Not implemented yet')
        else:
            raise ValueError('Invalid potential type')

    @staticmethod
    def init_dataset(config):
        if config['train']['dataset']['distribution'] == 'Gaussian':
            d = config['dim']
            return GaussianData(jnp.zeros((d,)), jnp.eye(d))
        else:
            raise ValueError('Invalid distribution type')
