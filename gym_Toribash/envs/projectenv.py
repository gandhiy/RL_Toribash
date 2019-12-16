# Custom gym environment built with insight from Tori.Env 
# in torille (https://github.com/Miffyli/ToriLLE)

# This is the environment that will be trained using stable-baselines.
# Once a robust environment has been built, stable-baselines allows for 
# many different RL algorithms to be trained around them. 

import gym 
import torille
import pickle
import math
import numpy as np 


from gym import spaces
from copy import deepcopy
from torille import constants
from utils.tools import *
from utils.rewards import *
from utils.action_space import bin_val
from torille.envs.gym_env import ToriEnv




class SingleAgentToribash(ToriEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reward_func_dict = {
            0: transition_constrained_reward,
            1: linreg_reward,
            2: dist_reward,
            3: curriculum_reward
        }
        self.reward_val = kwargs['reward_func']
        self.reward_function = self.reward_func_dict[self.reward_val]
        self.limbs = create_dictionary_mapper(True)
        self.joints = create_dictionary_mapper(False)
        self.discrete_action_space = False
        self.continuous = False

        self.reward_kwargs = {}
        self.reward_kwargs['iteration'] = 0
        self.reward_kwargs['terminal'] = False


        self.observation_space = spaces.Box(
            low=-50, high=50, dtype=np.float32,
            shape=((2*( (6 * constants.NUM_LIMBS) + constants.NUM_CONTROLLABLES + 1)),)
        )

    def init(self, **kwargs):
        """
         Initialize environment specific variables such as reward type 
         and action type. If action type is discrete, creates a dictionary
         from model output (int) to environment action ({1,2,3,4}^22). If the 
         action is continuous, builds up a grid to learn the continuous values 
         from. 
        """

        # setting up action 
        if(kwargs['action_kwargs']['type'] == 'discrete'):
            self.discrete_action_space = True
            self.action_dict = kwargs['action_dictionary']
            self.action_space = spaces.Discrete(len(self.action_dict))
        elif(kwargs['action_kwargs']['type'] == 'continuous'):
            # continuous action space
            self.continuous = True
            # define grid size
            self.T = kwargs['action_kwargs']['high'] - kwargs['action_kwargs']['low']
            self.action_space = spaces.Box(kwargs['action_kwargs']['low'], kwargs['action_kwargs']['high'], shape=(2,))
        else:
            # learn from the full action space
            self.action_space = spaces.MultiDiscrete([constants.NUM_JOINT_STATES] * constants.NUM_CONTROLLABLES)      

        # setting up reward    
        self.reward_kwargs['timesteps'] = 256*math.ceil(kwargs['timesteps']/256)
        self.reward_kwargs.update(kwargs['reward_kwargs'])


    def _preprocess_observation(self, state):
        p1, p2 = process_state(state, self.limbs, self.joints)
        p1 = np.array(list(p1.items()))[:,1]
        p2 = np.array(list(p2.items()))[:,1]
        return np.concatenate((p1,p2), axis=0)

    def _preprocess_action(self, action):
        """
         process the action sent by self.step 

         Args:
            action (list): the action from stable-baselines (22,1) in {0,1,2,3}
         Return:
            action (list<list>): action to send to torille.ToribashControl.make_action.
            Must be (22,2) in {1,2,3,4}
        """

        # processing discrete action spaces is just indexing a dictionary
        if(self.discrete_action_space):
            action = self.action_dict[action]

        # turn (x,y) coordinate into action
        elif(self.continuous):
            base = int(bin_val(action[0], action[1], self.T))
            tmp = [int(c) + 1 for c in np.base_repr(base, base=4)]
            action = tmp + [1]*(22 - len(tmp))

        # a full action space learner
        else:
            if type(action) is not list:
                action = list(action)
            action = [a + 1 for a in action] 

        #add 2nd player (dummy) action
        action = [action, [3] * constants.NUM_CONTROLLABLES]
        # turn off 2nd player grips
        action[1][-2] = 1
        action[1][-1] = 1

        return action

    def _reward_function(self, old_state, new_state):
        return self.reward_function(
            old_state, 
            new_state, 
            self.limbs, 
            self.joints, 
            **self.reward_kwargs
        )

    def reset(self):
        """
         Reset the environment and return initial state observations

         Returns:
         obs (spaces.Box): the observations defined by _preprocess_observation
        """

        obs = None
        
        if self.just_created:
            # Initialize game here to make it pickable before init
            self.game.init()
            # The env was just created and we need to start
            # by returning an observation.
            state, terminal = self.game.get_state()
            obs = self._preprocess_observation(state)

            # Remember to update the state here as well
            self.old_state = state
            self.just_created = False
        else:
            # Reset episode
            state = self.game.reset()
            # Get the state and return it
            obs = self._preprocess_observation(state)
            self.old_state = state

        return obs

    def step(self, action):
        """
         Take a step in the environment (ToribashControl) and return the information about the 
         step

         Args:
            action (spaces.MultiDiscrete): the action for player 1 to take in the 
            environment
         Return:
            obs (spaces.Box): the observations defined by _preprocess_observation 
            from the state after taking an action
            reward (float): the reward value defined by old_state and new_state in _reward_function
            terminal (bool): whether the episode has ended
            info (dict): information to pass to the monitor
        """
        if self.just_created:
            raise Exception(
                "`step` function was called " +
                "before calling `reset`. Call `reset` after creating " +
                "environment to get the first observation."
            )
        
        # process action
        action = deepcopy(action)
        action = self._preprocess_action(action)
        self.game.make_actions(action)
        self.reward_kwargs['iteration'] += 1 # keep count of the iteration for curriculum rewards

        # step in the environment and get the reward
        state, terminal = self.game.get_state()
        self.reward_kwargs['terminal'] = terminal
        reward = self._reward_function(self.old_state, state)
        self.old_state = state
        
        


        obs = self._preprocess_observation(state)
        info = {'state': state}
        
        return obs, reward, terminal, info

