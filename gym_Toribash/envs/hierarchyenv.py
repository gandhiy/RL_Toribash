import gym
import math
import torille
import pickle
import numpy as np 

from gym import spaces
from copy import deepcopy
from torille import constants
from itertools import product
from utils.tools import *
from utils.rewards import *
from stable_baselines import PPO1
from utils.action_space import bin_val  
from torille.envs.gym_env import ToriEnv



class BaseLevel(ToriEnv):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.reward_function_dict = {
            0: score_only_reward,
            1: transition_constrained_reward,
            2: linreg_reward,
            3: curriculum_reward
        }
        self.reward_val = kwargs['reward_func']
        self.reward_function = self.reward_function_dict[self.reward_val]
        self.limbs = create_dictionary_mapper()
        self.joints = create_dictionary_mapper(False)
        
        self.reward_kwargs = {}
        self.reward_kwargs['iteration'] = 0
        self.reward_kwargs['terminal'] = False
        
        self.dummy_action = [4] * constants.NUM_CONTROLLABLES
        self.observation_space = spaces.Box(
            low = -50, high = 50, dtype=np.float32,
            shape=((2*( (6 * constants.NUM_LIMBS) + constants.NUM_CONTROLLABLES + 1)),)
        )
        

    def init(self, **kwargs):

        if(kwargs['action_kwargs']['type'] == 'discrete'):
            self.action_dict = kwargs['action_dictionary']
            self.action_space = spaces.Discrete(len(self.action_dict))
            self.type = 'disc'
        elif(kwargs['action_kwargs']['type'] == 'continuous'):
            self.T = kwargs['action_kwargs']['high'] - kwargs['action_kwargs']['low']
            self.action_space = spaces.Box(
                kwargs['action_kwargs']['low'], 
                kwargs['action_kwargs']['high'], shape=(2,))
            self.type = 'cont'
        else:
            self.action_space = spaces.MultiDiscrete(
                [constants.NUM_JOINT_STATES] * len(self.action_idx))
            self.type = 'full'

                
        
        self.reward_kwargs['timesteps'] = 256*math.ceil(kwargs['timesteps']/256)
        self.reward_kwargs.update(kwargs['reward_kwargs'])
        

    def _preprocess_observation(self, state):
        p1, p2 = process_state(state, self.limbs, self.joints)
        p1 = np.array(list(p1.items()))[:,1]
        p2 = np.array(list(p2.items()))[:,1]
        return np.concatenate((p1,p2), axis=0)

    def _preprocess_action(self, action):

        # tmp should be the same length action_idx
        if(self.type is 'disc'):
            tmp = self.action_dict[action]
        elif(self.type is 'cont'):
            base = int(bin_val(action[0], action[1], self.T, len(self.action_idx)))
            tmp = [int(c) + 1 for c in np.base_repr(base,base=4)]
            tmp = tmp + [1]*(len(self.action_idx) - len(tmp))
        elif(self.type is 'full'):
            if type(action) is not list:
                action = list(action)
            tmp = [a + 1 for a in action]
        return tmp

    def _preprocess_other_action(self, action, configs, action_idx):
        if configs['action_kwargs']['type'] == 'discrete':
            tmp = configs['action_dictionary'][action]
        elif configs['action_kwargs']['type'] == 'continuous':
            T = configs['action_kwargs']['high'] - configs['action_kwargs']['low']
            base = int(bin_val(action[0], action[1], T, len(action_idx)))
            tmp = [int(c) + 1 for c in np.base_repr(base, base=4)]
            tmp = tmp + [1] *(len(action_idx) - len(tmp))
        elif configs['action_kwargs']['type'] == 'full':
            if type(action) is not list:
                action = list(action)
            tmp = [a + 1 for a in action]
        return tmp
    
    def _reward_function(self, old_state, new_state):
        return self.reward_function(
            old_state, 
            new_state, 
            self.limbs, 
            self.joints, 
            **self.reward_kwargs
        )

    def reset(self):
        obs = None
        if(self.just_created):
            self.game.init()
            state, terminal = self.game.get_state()
            obs = self._preprocess_observation(state)
            self.old_state = state
            self.just_created = False
        else:
            state = self.game.reset()
            obs = self._preprocess_observation(state)
            self.old_state = state

        return obs

    def step(self, action):
        raise NotImplementedError


class MajorActions(BaseLevel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self,**kwargs):
        self.action_idx = [1, 3, 4, 5, 7, 8, 14, 15]
        super().init(**kwargs)
        
    def step(self,action):
        if self.just_created:
            raise Exception("no reset called")
        
        tmp = self._preprocess_action(action)
        
        action = deepcopy(self.dummy_action)
        for i,idx in enumerate(self.action_idx):
            action[idx] = tmp[i]
        

        action = [action, [3]*constants.NUM_CONTROLLABLES]
        action[0][-2] = 1
        action[0][-1] = 1
        action[1][-2] = 1
        action[1][-1] = 1

        self.game.make_actions(action)
        self.reward_kwargs['iteration'] += 1

        state, terminal = self.game.get_state()
        self.reward_kwargs['terminal'] = terminal
        reward = self._reward_function(self.old_state, state)
        self.old_state = state

        obs = self._preprocess_observation(state)
        info = {'state': state}

        return obs, reward, terminal, info


# In an alternate universe I would use inheritance, but for now, 
# just loading each model within the environment will have to do. 
class MinorActions(BaseLevel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self,**kwargs):
        self.action_idx = [0, 2, 6, 9, 12, 13, 16, 17]
        super().init(**kwargs)

        # the major actions        
        self.major = PPO1.load(kwargs['inner_models']['major_model'])
        with open(kwargs['inner_models']['major_model_configs'], 'rb') as f:
            self.major_configs = pickle.load(f)
        
        self.major_action_idx = [1, 3, 4, 5, 7, 8, 14, 15]
    
    
    def step(self, action):
        if self.just_created:
            raise Exception("no reset called")
        
        # predict the major action
        major_model_action, _  = self.major.predict(self._preprocess_observation(self.old_state))
        major_action = self._preprocess_other_action(major_model_action, self.major_configs, self.major_action_idx)

        
        # process the minor actions
        minor_action = self._preprocess_action(action)
        action = deepcopy(self.dummy_action)
        
        
        # combine major and minor actions
        for i,idx in enumerate(self.action_idx):
            action[idx] = minor_action[i]
        
        for i, idx in enumerate(self.major_action_idx):
            action[idx] = major_action[i]


        # build second player's actions and turn off grips
        action = [action, [3]*constants.NUM_CONTROLLABLES]
        action[0][-2] = 1
        action[0][-1] = 1
        action[1][-2] = 1
        action[1][-1] = 1
        
        
        # step through the game
        self.game.make_actions(action)
        self.reward_kwargs['iteration'] += 1



        # get the new state and update information
        state, terminal = self.game.get_state()
        self.reward_kwargs['terminal'] = terminal

        reward = self._reward_function(self.old_state, state)
        self.old_state = state 
        obs = self._preprocess_observation(state)
        info = {'state': state}

        return obs, reward, terminal, info


class Details(BaseLevel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self, **kwargs):
        self.action_idx = [10, 11, 18, 19, 20, 21]
        super().init(**kwargs)
        
        models = kwargs['inner_models']
        # set up major model and environment parameters
        self.major = PPO1.load(models['major_model'])
        with open(models['major_model_configs'], 'rb') as f:
            self.major_configs = pickle.load(f)        
        self.major_action_idx = [1, 3, 4, 5, 7, 8, 14, 15]

        # set up minor model and environment parameters
        self.minor = PPO1.load(models['minor_model'])
        with open(models['minor_model_configs'], 'rb') as f:
            self.minor_configs = pickle.load(f)
        self.minor_action_idx = [0, 2, 6, 9, 12, 13, 16, 17]

        
    def step(self, action):
        if self.just_created:
            raise Exception("no reset called")
        
        # predict the major action
        major_model_action, _  = self.major.predict(self._preprocess_observation(self.old_state))
        major_action = self._preprocess_other_action(major_model_action, self.major_configs, self.major_action_idx)
        # predict the minor action
        minor_model_action, _ = self.minor.predict(self._preprocess_observation(self.old_state))
        minor_action = self._preprocess_other_action(minor_model_action, self.minor_configs, self.minor_action_idx)
        
        # process the detailed actions
        detail_action = self._preprocess_action(action)
        action = deepcopy(self.dummy_action)
        
        
        # combine all detailed, minor, and major actions
        for i,idx in enumerate(self.action_idx):
            action[idx] = detail_action[i]
        
        for i, idx in enumerate(self.minor_action_idx):
            action[idx] = minor_action[i]

        for i, idx in enumerate(self.major_action_idx):
            action[idx] = major_action[i]


        # build second player's actions
        action = [action, [3]*constants.NUM_CONTROLLABLES]
        action[1][-2] = 1
        action[1][-1] = 1
        
        
        # step through the game
        self.game.make_actions(action)
        self.reward_kwargs['iteration'] += 1


        # get the new state and update information
        state, terminal = self.game.get_state()
        self.reward_kwargs['terminal'] = terminal

        reward = self._reward_function(self.old_state, state)
        self.old_state = state 
        obs = self._preprocess_observation(state)
        info = {'state': state}

        return obs, reward, terminal, info

        

