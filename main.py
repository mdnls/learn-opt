import yaml
from runner import Runner
from nets import ReLU_MLP
from jax import numpy as jnp, random, tree_util

if __name__ == "__main__":
    '''
    1. load a hard coded config file (later will be cmd argument)
    2. create a dict out of it and pass to training method
    3. training method
        - input: config dict
        - output: params of trained model
        - save the output to disk 
    '''

    with open('configs/M-L5-D2-B.yml') as f_in:
        config = yaml.safe_load(f_in)

    R = Runner(config)
    seed = random.PRNGKey(3621)
    model, potential, dataset = R.init_model(config, seed), R.init_potential(config), R.init_dataset(config)

    trained_model = R.optimize_model(config, seed, model, potential, dataset)
    print(config)