import gym
import torille
import pickle
import numpy as np 


from gym import spaces 
from copy import deepcopy
from torille import constants
from itertools import product
from utils.tools import *
from utils.lower_limbs_rewards import *
from torille.envs.gym_env import ToriEnv



class BodyEnv(ToriEnv):
    """
     Intermediate Class type for lower level environments
    """
    def __init__(self, **kwargs):
        """
         Initialize the body environment 
        """
        super().__init__(**kwargs)
        self.limbs_dict = create_dictionary_mapper(True)
        self.joint_dict = create_dictionary_mapper(False)

        # each body part can have a different reward function
        self.reward_dictionary = {
            0: left_leg_reward,
            1: right_leg_reward,
            2: left_arm_reward,
            3: right_arm_reward,
            4: upper_body_reward,
            5: lower_body_reward,
        }

        # Run each part individually to train or as 
        # components of the hierarchical model
        # self.individual = False
        self.reward_kwargs = {}
        self.reward_kwargs['trajectory'] = []

        #dummy action
        self.game_action =  [3] * constants.NUM_CONTROLLABLES
        # turn off player one grip initially, 
        # grip can be turned back on through model
        self.game_action[-2] = 1
        self.game_action[-1] = 1
        
        
    
    def init(self, **kwargs):
        """
         Initialize the environment using arguments from 
         the configuration file

         Args:
            kwargs: configurations from the yaml file
        """
        # the kwargs from config file
        self.reward_kwargs.update(kwargs['reward_kwargs'])

        # self.individual = kwargs['reward_kwargs']
        # observation space (the higher level signal)
        self.observation_space = spaces.Box(
            low=-1, high=1, dtype=np.float32,
            shape=(4,) 
        )
    

    def _preprocess_observation(self, state):
        """
         Return a random signal for training an individual 
         component of the body

         Args:
            state (numpy.ndarray): the signal from the higher level
        """

        # random signal
        # because the covariance matrix must be semidefinite,
        # the last two values are absolute valued.
        signal = 2*np.random.rand(4) - 1
        signal[2] = abs(signal[2])
        signal[3] = abs(signal[3])
        return signal

    def reset(self):
        """
         Reset the environment after the end of an episode

        """

        # reset the episode trajectory
        self.reward_kwargs['trajectory'] = []
        obs = None


        if self.just_created:
            self.game.init()
            state, terminal = self.game.get_state()
            obs = self._preprocess_observation(state)
            self.old_state = state
            self.just_created = False

        else:
            state = self.game.reset()
            obs = self._preprocess_observation(state)
            self.old_state = state

        # add state s0 to the trajectory
        p1, _ = process_state(state, self.limbs_dict, self.joint_dict)
        self.reward_kwargs['trajectory'].append([p1[k] for k in p1.keys() if 'act' in k])

        return obs
    
    def step(self, action):
        """
         Step through the environment. Is not used in the 
         hierarchy model, so defines stepping through an environment
         with an individual body component

         Args:
            action (int): the output index with the highest softmax 
            score. 

         
        """
        if self.just_created:
            raise Exception(
                "`step` function was called " +
                "before calling `reset`. Call `reset` after creating " +
                "environment to get the first observation."
            )

        # the action index is necessary for the 
        # reward function
        self.reward_kwargs['action_idx'] = action

        # process the discrete action and take 
        # action in the environment
        action = self._preprocess_action(action)
        self.game.make_actions(action)

        # get the state
        state, terminal = self.game.get_state()
        
        # the observations that the model gets are the 
        # signals from the higher level so the 
        # _process_observation function ignores the 
        # state from the environment
        obs = self._preprocess_observation(state)
        self.reward_kwargs['signal'] = obs
        
        # add trajectory to reward kwargs
        p1, _ = process_state(state, self.limbs_dict, self.joint_dict)
        self.reward_kwargs['trajectory'].append([p1[k] for k in p1.keys() if 'act' in k])
        
        # get the reward from the state
        reward = self._reward_function(self.old_state, state)
        
        # set the old state to current state
        self.old_state = state

        # empty info 
        info = {}
        return obs, reward, terminal, info 



class Left_Leg(BodyEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # initialize list of actions to choose from
        self.action_list = list(product([1,2,3,4], repeat=3))

        # set action space
        self.action_space = spaces.Discrete(len(self.action_list))

        # number of actions is used in the reward function
        self.reward_kwargs['number_actions'] = len(self.action_list)
        self.reward_kwargs['controllable_actions'] = 3

    def init(self, **kwargs):
        # initialize with config kwargs
        super().init(**kwargs)

    def get_action_part(self, idx):
        """
         Makes getting the action for the upper levels
         a bit easier

         Args:
            idx (int): the action index predicted by the model
        """
        # which action from the action list 
        a = list(self.action_list[idx]) 
        # which components of the dummy game to change
        # hard-coded because the values don't change
        action_idx = [15,17,19]
        return (a, action_idx)

    def _preprocess_action(self, idx):
        """
         Process the value chosen by the model. The model returns
         a integer value from [0, 4**N) where N is the number of joints
         controlled by the specific model. This index value is processed
         to return an action interpretable by the game environment

         Args:
            idx (int): the value returned by the model
        """
        a, action_idx = self.get_action_part(idx)
        single_action = deepcopy(self.game_action)
        single_action[action_idx[0]] = a[0]
        single_action[action_idx[1]] = a[1]
        single_action[action_idx[2]] = a[2]
        
        action = [single_action, [3]*constants.NUM_CONTROLLABLES]
        # turn off dummy grip for player 2
        action[1][-2] = 1
        action[1][-1] = 1
        return action

    def _reward_function(self, old_state, new_state):
        """
         Reward function defined in utils.lower_limbs_rewards
        """
        return self.reward_dictionary[0](
            old_state, 
            new_state, 
            self.limbs_dict, 
            self.joint_dict, 
            **self.reward_kwargs
        )


"""
 Differences between each components:
 1.) The number of joints controlled 
 2.) The action_idx list in get_action_part
 3.) The reward function being called (some redundancies here, for safety)
"""

class Right_Leg(BodyEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # initialize list of actions to choose from
        self.action_list = list(product([1,2,3,4], repeat=3))

        # set action space
        self.action_space = spaces.Discrete(len(self.action_list))

        # number of actions is used in the reward function
        self.reward_kwargs['number_actions'] = len(self.action_list)
        self.reward_kwargs['controllable_actions'] = 3
        
    def init(self, **kwargs):
        """
         Set up body specific reward arguments
        """
        super().init(**kwargs)
        
    def get_action_part(self, idx):
        """
         Makes getting the action for the upper levels
         a bit easier

         Args:
            idx (int): the action index predicted by the model
        """
        # which action from the action list 
        a = list(self.action_list[idx]) 
        # which components of the dummy game to change
        # hard-coded because the values don't change
        action_idx = [14,16,18]
        return (a, action_idx)


    def _preprocess_action(self, idx):
        """
         Process the value chosen by the model. The model returns
         a integer value from [0, 4**N) where N is the number of joints
         controlled by the specific model. This index value is processed
         to return an action interpretable by the game environment

         Args:
            idx (int): the value returned by the model
        """
        a, action_idx = self.get_action_part(idx)
        single_action = deepcopy(self.game_action)
        single_action[action_idx[0]] = a[0]
        single_action[action_idx[1]] = a[1]
        single_action[action_idx[2]] = a[2]
        
        action = [single_action, [3]*constants.NUM_CONTROLLABLES]
        # turn off dummy grip for player 2
        action[1][-2] = 1
        action[1][-1] = 1
        return action

    def _reward_function(self, old_state, new_state):
        """
         Reward function defined in utils.hierarchy_rewards
        """
        return self.reward_dictionary[1](
            old_state, 
            new_state, 
            self.limbs_dict, 
            self.joint_dict, 
            **self.reward_kwargs
        )

class Left_Arm(BodyEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # initialize list of actions to choose from
        self.action_list = list(product([1,2,3,4], repeat=4))

        # set action space
        self.action_space = spaces.Discrete(len(self.action_list))

        # number of actions is used in the reward function
        self.reward_kwargs['number_actions'] = len(self.action_list)
        self.reward_kwargs['controllable_actions'] = 4

    def init(self, **kwargs):
        """
         Set up body specific reward arguments
        """
        super().init(**kwargs)
        
    def get_action_part(self, idx):
        """
         Makes getting the action for the upper levels
         a bit easier

         Args:
            idx (int): the action index predicted by the model
        """
        # which action from the action list 
        a = list(self.action_list[idx]) 
        # which components of the dummy game to change
        # hard-coded because the values don't change
        action_idx = [8,9,11,21]
        return (a, action_idx)
    
    def _preprocess_action(self, idx):
        """
         Process the value chosen by the model. The model returns
         a integer value from [0, 4**N) where N is the number of joints
         controlled by the specific model. This index value is processed
         to return an action interpretable by the game environment

         Args:
            idx (int): the value returned by the model
        """
        a, action_idx = self.get_action_part(idx)
        single_action = deepcopy(self.game_action)
        single_action[action_idx[0]] = a[0]
        single_action[action_idx[1]] = a[1]
        single_action[action_idx[2]] = a[2]
        single_action[action_idx[3]] = a[3]
        
        action = [single_action, [3]*constants.NUM_CONTROLLABLES]
        # turn off dummy grip for player 2
        action[1][-2] = 1
        action[1][-1] = 1
        return action

    def _reward_function(self, old_state, new_state):
        """
         Reward function defined in utils.hierarchy_rewards
        """
        return self.reward_dictionary[2](
            old_state, 
            new_state, 
            self.limbs_dict, 
            self.joint_dict, 
            **self.reward_kwargs
        )

class Right_Arm(BodyEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # initialize list of actions to choose from
        self.action_list = list(product([1,2,3,4], repeat=4))

        # set action space
        self.action_space = spaces.Discrete(len(self.action_list))

        # number of actions is used in the reward function
        self.reward_kwargs['number_actions'] = len(self.action_list)
        self.reward_kwargs['controllable_actions'] = 4


    def init(self, **kwargs):
        """
         Set up body specific reward arguments
        """
        super().init(**kwargs)
        
    def get_action_part(self, idx):
        """
         Makes getting the action for the upper levels
         a bit easier

         Args:
            idx (int): the action index predicted by the model
        """
        # which action from the action list 
        a = list(self.action_list[idx]) 
        # which components of the dummy game to change
        # hard-coded because the values don't change
        action_idx = [5,6,10,20]
        return (a, action_idx)

    def _preprocess_action(self, idx):
        """
         Process the value chosen by the model. The model returns
         a integer value from [0, 4**N) where N is the number of joints
         controlled by the specific model. This index value is processed
         to return an action interpretable by the game environment

         Args:
            idx (int): the value returned by the model
        """
        a, action_idx = self.get_action_part(idx)
        single_action = deepcopy(self.game_action)
        single_action[action_idx[0]] = a[0]
        single_action[action_idx[1]] = a[1]
        single_action[action_idx[2]] = a[2]
        single_action[action_idx[3]] = a[3]
        
        action = [single_action, [3]*constants.NUM_CONTROLLABLES]
        # turn off dummy grip for player 2
        action[1][-2] = 1
        action[1][-1] = 1
        return action

    def _reward_function(self, old_state, new_state):
        """
         Reward function defined in utils.hierarchy_rewards
        """
        return self.reward_dictionary[3](
            old_state, 
            new_state, 
            self.limbs_dict, 
            self.joint_dict, 
            **self.reward_kwargs
        )

class Upper_Body(BodyEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # initialize list of actions to choose from
        self.action_list = list(product([1,2,3,4], repeat=4))

        # set action space
        self.action_space = spaces.Discrete(len(self.action_list))

        # number of actions is used in the reward function
        self.reward_kwargs['number_actions'] = len(self.action_list)
        self.reward_kwargs['controllable_actions'] = 4

    def init(self, **kwargs):
        """
         Set up body specific reward arguments
        """
        super().init(**kwargs)
    

    def get_action_part(self, idx):
        """
         Makes getting the action for the upper levels
         a bit easier

         Args:
            idx (int): the action index predicted by the model
        """
        # which action from the action list 
        a = list(self.action_list[idx]) 
        # which components of the dummy game to change
        # hard-coded because the values don't change
        action_idx = [0,1,4,7]
        return (a, action_idx)

    def _preprocess_action(self, idx):
        """
         Process the value chosen by the model. The model returns
         a integer value from [0, 4**N) where N is the number of joints
         controlled by the specific model. This index value is processed
         to return an action interpretable by the game environment

         Args:
            idx (int): the value returned by the model
        """
        a, action_idx = self.get_action_part(idx)
        single_action = deepcopy(self.game_action)
        single_action[action_idx[0]] = a[0]
        single_action[action_idx[1]] = a[1]
        single_action[action_idx[2]] = a[2]
        single_action[action_idx[3]] = a[3]
        
        action = [single_action, [3]*constants.NUM_CONTROLLABLES]
        # turn off dummy grip for player 2
        action[1][-2] = 1
        action[1][-1] = 1
        return action

    def _reward_function(self, old_state, new_state):
        """
         Reward function defined in utils.hierarchy_rewards
        """
        return self.reward_dictionary[4](
            old_state, 
            new_state, 
            self.limbs_dict, 
            self.joint_dict, 
            **self.reward_kwargs
        )

class Lower_Body(BodyEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # initialize list of actions to choose from
        self.action_list = list(product([1,2,3,4], repeat=4))

        # set action space
        self.action_space = spaces.Discrete(len(self.action_list))

        # number of actions is used in the reward function
        self.reward_kwargs['number_actions'] = len(self.action_list)
        self.reward_kwargs['controllable_actions'] = 4

    def init(self, **kwargs):
        """
         Set up body specific reward arguments
        """
        super().init(**kwargs)

    def get_action_part(self, idx):
        """
         Makes getting the action for the upper levels
         a bit easier

         Args:
            idx (int): the action index predicted by the model
        """
        # which action from the action list 
        a = list(self.action_list[idx]) 
        # which components of the dummy game to change
        # hard-coded because the values don't change
        action_idx = [2,3,12,13]
        return (a, action_idx)
        

    def _preprocess_action(self, idx):
        """
         Process the value chosen by the model. The model returns
         an integer value from [0, 4**N) where N is the number of joints
         controlled by the specific model. This index value is processed
         to return an action interpretable by the game environment

         Args:
            idx (int): the value returned by the model
        """
        a, action_idx = self.get_action_part(idx)
        single_action = deepcopy(self.game_action)
        single_action[action_idx[0]] = a[0]
        single_action[action_idx[1]] = a[1]
        single_action[action_idx[2]] = a[2]
        single_action[action_idx[3]] = a[3]
        
        action = [single_action, [3]*constants.NUM_CONTROLLABLES]
        # turn off grip for player 2
        action[1][-2] = 1
        action[1][-1] = 1
        return action

    def _reward_function(self, old_state, new_state):
        """
         Reward function defined in utils.hierarchy_rewards
        """
        return self.reward_dictionary[5](
            old_state, 
            new_state, 
            self.limbs_dict, 
            self.joint_dict, 
            **self.reward_kwargs
        )