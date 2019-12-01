import os 
import gym
import pickle
import gym_Toribash



from yaml import load, dump
from stable_baselines import PPO1
from argparse import ArgumentParser
from utils.train_Ridge import train_ridge
from utils.action_space import action_space
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


def check_action_space(env_dict):
    """
     Check how the action space is suppose to be built
    """
    # using a discrete set of actions
    if(env_dict['action_kwargs']['type'] == 'discrete'):
        # action space configurations
        N = env_dict['action_kwargs']['num_actions']
        builder = action_space(env_dict['player_1_csv'])

        # which action space to use
        if(env_dict['action_kwargs']['method'] == 'weighted'):
            action_dict = builder.temporal_frequency_builder(N)
        elif(env_dict['action_kwargs']['method'] == 'random'):
            action_dict = builder.random_actions(N)
        else:
            action_dict = builder.frequency_builder(N)
        
        # add new dictionary to environment
        env_dict['action_dictionary'] = action_dict
    
    return env_dict

def check_reward_space(env_dict):
    """
     Check how the reward space is suppose to be built
    """
    if(env_dict['reward'] >= 2):
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

def train(env_dict):
    """
     Run training on a Toribash Environment. Saves a model and the environment 
     configurations used. Because the actions may need to be remembered, this 
     method builds the action space here and saves it to the environment dictionary

     Args:
        env_dict (dictionary): The dictionary from the yaml file. 
        Please refer to the github page for more information
    """
    # setting up reward and action space
    env_dict = check_reward_space(check_action_space(env_dict))
    env_name = "Toribash-SingleAgentToribash-v{}".format(env_dict['reward'])    
    env_dict['env_name'] = env_name

    # setting up the model and environment
    env = gym.make(env_name)
    env.init(**env_dict)    
    env = DummyVecEnv([lambda: env])
    model = PPO1(MlpPolicy, env, verbose=1)

    # learning the model
    try:
        model.learn(total_timesteps=env_dict['timesteps'])
    except KeyboardInterrupt as identifier:
        print("Incomplete Model Save")
        model.save('models/{}'.format(env_dict['savename']+"_incomplete"))
    finally:
        model.save('models/{}'.format(env_dict['savename']))

    # save these configurations to deploy model later
    # new fields have likely been added
    with open("models/{}_configs_dict.pkl".format(env_dict['savename']), 'wb') as f:
        pickle.dump(env_dict, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
    help='Training configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        env_dict = load(f)['env_dict']
    train(env_dict)
    

