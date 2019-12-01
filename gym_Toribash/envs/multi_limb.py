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
from torille.envs.gym_env import ToriEnv



class subModel:
    def __init__(self, model, action_idx):
        """
         A simple class wrapper for lower level models. 
         Has some useful functions for learning with.

         Args:
             model (string): path to the model
             action_idx (list): the list of action indexes controlled by the model
        """
        self.model = PPO1.load(model)
        self.action_indexes = action_idx
        self.action_list = list(product([1,2,3,4], repeat=len(self.action_indexes)))

    def predict(self,obs):
        """
         Run prediction on the model using 
         the signal sent by the higher level

         Args:
            obs (numpy.ndarray): the output from the higher level network
        """
        action, _ =  self.model.predict(obs)
        return self.action_list[action]
    

    

class MultiLimbToribash(ToriEnv):

    def __init__(self, **kwargs):
        """
         A environment controller that uses 6 different models to 
         control different components of the body rather than a 
         single model for the entire body.
        """
        super().__init__(**kwargs)

        self.reward_function_dict = {
         0: score_only_reward,
         1: transition_constrained_reward,
         2: linreg_reward
        }
        self.reward_val = kwargs['reward_func']
        self.reward_function = self.reward_function_dict[self.reward_val]
        self.limbs_dict = create_dictionary_mapper(True)
        self.joint_dict = create_dictionary_mapper(False)
        self.reward_kwargs = {}
        self.reward_kwargs['iteration'] = 0
        self.reward_kwargs['terminal'] = False
        self.dummy_action = [3] * constants.NUM_CONTROLLABLES

        # full state space as the observations
        self.observation_space = spaces.Box(
            low=-50, high=50, dtype=np.float32,
            shape=((2*( (6 * constants.NUM_LIMBS) + constants.NUM_CONTROLLABLES + 1)),)
        )

        # action space is the signal sent to each of the models
        self.action_space = spaces.Box(
            low = -1, high=1, dtype=np.float32,
            shape=(4,)
        )
        
    def create_component_list(self,**kwargs):
        """
         Add all components of the lower levels 
         to a list to be used by other methods. 
        """
        self.components = []
        self.components.append(subModel(kwargs['left_leg_model'], [15,17,19]))
        self.components.append(subModel(kwargs['right_leg_model'], [14,16,18]))
        self.components.append(subModel(kwargs['left_arm_model'], [8,9,11,21]))
        self.components.append(subModel(kwargs['right_arm_model'], [5,6,10,20]))
        self.components.append(subModel(kwargs['upper_body_model'], [0,1,4,7]))
        self.components.append(subModel(kwargs['lower_body_model'], [2,3,12,13]))
    
    def init(self, **kwargs):
        """
         Initialize environment specific variables. 
        """

        # set up reward function args
        self.reward_kwargs.update(kwargs['reward_kwargs'])
        self.reward_kwargs['timesteps'] = kwargs['timesteps']
        # set up lower level models 
        self.create_component_list(**kwargs)

    def _preprocess_action(self, signal):
        """
         Process the value chosen by the model. The model returns 
         4 continuous variables from [-1, 1]. This signal is then 
         sent to the lower levels which are trained to predict actions
         from a distribution defined by that signal for their respective
         joints. These actions are combined to form the full action

         Args: 
            signal (numpy.ndarray): model output
        """
        player_1_action = deepcopy(self.dummy_action)


        for comp in self.components:
            act = comp.predict(signal)
            for i,idx in enumerate(comp.action_indexes):
                player_1_action[idx] = act[i]
        action = [player_1_action, deepcopy(self.dummy_action)]

        # turn off grip for player 2
        action[1][-2] = 1
        action[1][-1] = 1

        return action
                
    def _preprocess_observation(self, state):
        p1, p2 = process_state(state, self.limbs_dict, self.joint_dict)
        p1 = np.array(list(p1.items()))[:,1]
        p2 = np.array(list(p2.items()))[:,1]
        return np.concatenate((p1,p2), axis=0)
    
    def _reward_function(self, old_state, new_state):
        return self.reward_function(
            old_state,
            new_state,
            self.limbs_dict,
            self.joint_dict,
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

            NOTE: Had to be slightly rewritten to include info

            """
            if self.just_created:
                raise Exception(
                    "`step` function was called " +
                    "before calling `reset`. Call `reset` after creating " +
                    "environment to get the first observation."
                )
            action = deepcopy(action)
            action = self._preprocess_action(action)
            self.game.make_actions(action)
            self.reward_kwargs['iterations'] += 1 # keep count of the learning iterations

            state, terminal = self.game.get_state()
            self.reward_kwargs['terminal'] = terminal
            reward = self._reward_function(self.old_state, state)
            self.old_state = state
            
            
            obs = self._preprocess_observation(state)
            info = {'state': state}
            
            return obs, reward, terminal, info

