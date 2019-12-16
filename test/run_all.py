"""
Running this will built a yaml file for all 
of the possible single agent tuns
"""

import os 
import sys 
import yaml

import tqdm
from os.path import join
from itertools import product

# all possible combinations
model_names = list(product(
    ["full", "random", "frequent", "weighted", "cont"],
    ['0', '1', '2', '3']
))

# constants for each reward
reward_kwargs = {
    0: {
        'c0': 1.2,
        'c1': 1.2,
        'c2': 0.4,
    },
    1: {
        'c0': 1.3,
        'c1': 0.9,
        'phi': 'data/reward_weights.pkl'

    },
    2: {
        'c0': 0.2,
        'c1': 1.9,
    },
    3: {
        'c0': 0.5,
        'c1': 0.5,
        'c2': 1.0,
        'c3': 5.0,
        'c4': 0.25,
        'phi': 'data/reward_weights.pkl'
    }

}
action_kwargs= {
    'full':{
        'type': 'full',
    },
    'random':{
        'type': 'discrete',
        'method': 'random',
        'num_actions': 30,
    },
    'frequent':{
        'type': 'discrete',
        'method': 'frequent',
        'num_actions': 30,
    },
    'weighted':{
        'type': 'discrete',
        'method': 'weighted',
        'num_actions': 30,
    },
    'cont': {
        'type': 'continuous',
        'low': 0,
        'high': 10,
    }

}


base_dict = {
    'env_dict': {
        'agent': 'single',
        'player_1_csv': 'data/player1_state_info.csv',
        'player_2_csv': 'data/player1_state_info.csv',
        'timesteps': 100000,
        'reward':0,
        'reward_kwargs': None,
        'action_kwargs':None,
    }
}


for i in tqdm.trange(len(model_names)):
    base_dict['env_dict']['action_kwargs'] = action_kwargs[model_names[i][0]]
    base_dict['env_dict']['reward'] = int(model_names[i][1])
    base_dict['env_dict']['reward_kwargs'] = reward_kwargs[int(model_names[i][1])]
    name = '_'.join(model_names[i])
    base_dict['env_dict']['savename'] = name
    path = os.path.join('configs', name+"_configs.yml")
    with open(path, 'w') as f:
        yaml.safe_dump(base_dict, f)
    print(path)
    cmd = 'python model_trainer.py -c {}'.format(path)
    os.system(cmd)
