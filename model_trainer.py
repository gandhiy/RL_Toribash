import os 
import gym
import math
import pickle
import numpy as np
import gym_Toribash


from yaml import load, dump
from argparse import ArgumentParser
from utils.sto_sim import Stochastic_Similarity
from stable_baselines import PPO1
from utils.train_Ridge import train_ridge
from utils.action_space import action_space
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy


best_mean_reward = -np.inf
num_saves = 5
save_iter = 0

def callback(_locals, _globals):
    global best_mean_reward, num_saves, save_iter
    
    total_batches = math.ceil(_locals['total_timesteps']/256)
    if(total_batches < num_saves):
        num_saves = total_batches
    

    # # save intermediate runs
    if('iters_so_far' in _locals.keys()):
        if _locals['iters_so_far'] % (total_batches//num_saves) == 0:
            # save a model
            model_path = os.path.join(save_folder, 'model_{}_of_{}.pkl'.format(save_iter, num_saves))
            _locals['self'].save(model_path)
            save_iter += 1


        # save best run
        x,y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                _locals['self'].save(os.path.join(save_folder, 'best_model.pkl'))

    return True


# some helper functions to make it easier to read the main function
def load_actions(env_dict):
    """
     Check how the action space is suppose to be built.
     Not used for hierarchy models
    """
    if(env_dict['agent'] == 'hierarchy'):
        if(env_dict['level'] == 'major'):
            controllable = [1, 3, 4, 5, 7, 8, 14, 15]
        elif(env_dict['level'] == 'minor'):
            controllable = [0, 2, 6, 9, 12, 13, 16, 17]
        elif(env_dict['level'] == 'detail'):
            controllable = [10, 11, 18, 19, 20, 21]
    else:
        controllable = 22

    if(env_dict['action_kwargs']['type'] == 'discrete'):
        N = env_dict['action_kwargs']['num_actions']
        builder = action_space(env_dict['player_1_csv'])

        # which action space to use
        if(env_dict['action_kwargs']['method'] == 'weighted'):
            env_dict['action_dictionary'] = builder.temporal_frequency_builder(N, controllable)

        elif(env_dict['action_kwargs']['method'] == 'random'):
            env_dict['action_dictionary'] = builder.random_actions(N, controllable)

        elif(env_dict['action_kwargs']['method'] == 'frequent'):
            env_dict['action_dictionary'] = builder.frequency_builder(N, controllable)

        else:
            raise ValueError("Unknown action builder method. Please choose from [weighted, random, frequent]")


    return env_dict


def load_phi(env_dict):
    """
     Load in the reward weights either from a 
     given path in the configs or built with the 
     match csv files
    """

    if('phi' not in env_dict['reward_kwargs'].keys()):
        env_dict['reward_kwargs']['phi'] = train_ridge(
            env_dict['player_1_csv'], 
            env_dict['player_2_csv'], 
            'data/{}'.format(env_dict['savename']
        ))
    else:
        with open(env_dict['reward_kwargs']['phi'], 'rb') as f:
            env_dict['reward_kwargs']['phi'] = pickle.load(f)
        
    return env_dict


def check_limbs(env_dict):
    f = open('data/limbs.txt', 'r')
    needed_limbs = [l.rstrip() for l in f]
    limbs = env_dict['limb_models']
    
    for k,v in limbs.items():
        if(k not in needed_limbs):
            raise ValueError("Incorrect model name: {}".format(k))
        else:
            needed_limbs.remove(k)
    return needed_limbs
    

def make_env(env_dict, env_name):
    env = gym.make(env_name)
    env.settings.set("matchframes", 1000)
    env.init(**env_dict)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    return env


def load_single_model(env_dict):
    env_dict = load_actions(env_dict)
    env_dict = load_phi(env_dict)
    env_dict['env_name'] = "Toribash-SingleAgentToribash-v{}".format(env_dict['reward'])  
    return env_dict


def load_multi_model(env_dict):
    env_dict = load_phi(env_dict)
    
    # check for all models
    ls = check_limbs(env_dict)
    if(len(ls) > 0):
        raise ValueError(
        "Make sure there is a model for each of the limbs." +
        "\n Missing the following {}".format(ls))

    env_dict['env_name'] = 'Toribash-MultiLimb-v{}'.format(env_dict['reward'])
    return env_dict


def load_hierarchy_model(env_dict):
    env_dict = load_actions(env_dict)
    env_dict = load_phi(env_dict)
    if(env_dict['level'] == 'major'):
        env_dict['env_name'] = 'Toribash-MajorEnv-v{}'.format(env_dict['reward'])
    elif(env_dict['level'] == 'minor'):
        env_dict['env_name'] = 'Toribash-MinorEnv-v{}'.format(env_dict['reward'])
    elif(env_dict['level'] == 'detail'):
        env_dict['env_name'] = 'Toribash-DetailEnv-v{}'.format(env_dict['reward'])
    else:
        raise ValueError("Unknown hierarchy level. Please choose from [major, minor, detail]")
    
    return env_dict


def train(env_dict, save_folder, log_dir):
    """
     Run training on a Toribash Environment. Saves a model and the environment 
     configurations used. Because the actions may need to be remembered, this 
     method builds the action space here and saves it to the environment dictionary

     Args:
        env_dict (dictionary): The dictionary from the yaml file. 
        save_folder (filepath): path to save models
        log_dir (filepath): path to save logs. If file is run, then found inside of save_folder
    """


    # setting up reward and action space

    if(env_dict['agent'] == 'single'):
        env_dict = load_single_model(env_dict)
    elif(env_dict['agent'] == 'multi'):
        env_dict = load_multi_model(env_dict)
    elif(env_dict['agent'] == 'limb'):
        env_dict['env_name'] = 'Toribash-{}-v0'.format(env_dict['limb'])
    elif(env_dict['agent'] == 'hierarchy'):
        env_dict = load_hierarchy_model(env_dict)
    else:
        raise ValueError("Incorrect agent type given. Make sure agent: [single, multi, limb, hierarchy]" +
    "\n And, make sure other necessary components are loaded correctly."
    )

    with open(os.path.join(save_folder, 'configs_dict.pkl'), 'wb') as f:
        pickle.dump(env_dict, f)



    # setting up the model and environment
    env = make_env(env_dict, env_dict['env_name'])

    model = PPO1(MlpPolicy, env, verbose=1, tensorboard_log="./tensorboard/{}/".format(env_dict['savename']), optim_stepsize=0.01)

    try:
        model.learn(total_timesteps=env_dict['timesteps'], callback=callback)
    except KeyboardInterrupt as identifier:
        print("Incomplete Model Save")
        model.save(os.path.join(save_folder, 'incomplete'))
    finally:
        model.save(os.path.join(save_folder, 'final_model.pkl'))





if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
    help='Training configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        env_dict = load(f)['env_dict']
    
    save_folder = os.path.join("models/", env_dict['savename'])
    os.makedirs(save_folder, exist_ok=True)
    log_dir = os.path.join(save_folder, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    train(env_dict, save_folder, log_dir)
    

