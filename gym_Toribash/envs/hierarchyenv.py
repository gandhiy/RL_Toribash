import gym
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
        
        self.dummy_action = [1] * constants.NUM_CONTROLLABLES
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

                
        self.reward_kwargs.update(kwargs['reward_kwargs'])
        self.reward_kwargs['timesteps'] = kwargs['timesteps']

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
        super().init(**kwargs)
        self.action_idx = [1, 3, 5, 6, 8, 9, 12, 13, 16, 17]

    def step(self,action):
        if self.just_created:
            raise Exception("no reset called")
        tmp = self._preprocess_action(action)
        action = deepcopy(self.dummy_action)
        for i,idx in enumerate(self.action_idx):
            action[idx] = tmp[i]
        action = [action, [3]*constants.NUM_CONTROLLABLES]
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


class MinorActions(MajorActions):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self,**kwargs):
        super().init(**kwargs)
        self.action_idx = [0, 2, 4, 7, 11, 12, 14, 15, 18, 19]

        # the major actions        
        self.major = PPO1.load(kwargs['major_model'])
        self.major_action_idx = self.major.env.venv.envs[0].action_idx
    
    
    def step(self, action):
        if self.just_created:
            raise Exception("no reset called")
        
        #predict the major action
        pred, _  = self.major.predict(self._preprocess_observation(self.old_state))
        major_tmp = self.major.env.vec.envs[0]._preprocess_action(pred)

        # these are the minor actions
        tmp = self._preprocess_action(action)
        action = deepcopy(self.dummy_action)
        
        for i,idx in enumerate(self.action_idx):
            action[idx] = tmp[i]
        for j, jdx in enumerate(self.major_action_idx):
            action[jdx] = major_tmp[j]

        action = [action, [3]*constants.NUM_CONTROLLABLES]
        action[1][-2] = 1
        action[1][-1] = 1
        self.game.make_actions(action)

        state, terminal = self.game.get_state()
        reward = self._reward_function(self.old_state, state)
        self.old_state = state 

        obs = self._preprocess_observation(state)
        info = {}

        return obs, reward, terminal, info


class Grips(MinorActions):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self, **kwargs):
        super().init(**kwargs)
        self.action_idx = [20, 21]

        # the minor actions
        self.minor = PPO1.load(kwargs['minor_model'])
        

