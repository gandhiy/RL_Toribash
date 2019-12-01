import os 
import gym 
import gym_Toribash
import pickle


from yaml import load, dump
from argparse import ArgumentParser
from utils.train_Ridge import train_ridge
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1

def check_reward_space(env_dict):
    """
     Check how the reward space is suppose to be built
    """
    if(env_dict['reward'] == 2):
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
    # this is so the user can train a lower level model
    # or the hierarchy model given a reward function
    
    if(env_dict['env_name'] == 'None'):
        env_dict['env_name'] = 'Toribash-MultiLimb-v{}'.format(env_dict['reward'])
    env_dict = check_reward_space(env_dict)

    env = gym.make(env_dict['env_name'])
    env.init(**env_dict)

    env = DummyVecEnv([lambda: env])
    model = PPO1(MlpPolicy, env, verbose=1)

    try:
        model.learn(total_timesteps=env_dict['timesteps'])
    except KeyboardInterrupt as identifier:
        print("Incomplete Model Save")
        model.save('models/{}'.format(env_dict['savename']+"_incomplete"))
    finally:
        model.save('models/{}'.format(env_dict['savename']))

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