"""
 These reward functions are just for the limbs to learn 
 to correctly take a signal from the central controller. 

 Rather than have each limb try and maximize score in the game, 
"""

import numpy as np 

from scipy import stats
from utils.tools import process_state, load_csv


def reverse_bin(idx, N):
    """
     Gives the x and y points based on the action index number

     Args:
        idx (int): the index of which action to take 
        N (int): number of joints controlled by the action space
    """
    # discretization on each axis
    b = np.sqrt(4**N)
    x = 2*(idx%b)/b - 1 + 0.125 + 0.075*np.random.randn()
    y = 2*(idx//b)/b - 1 + 0.125 + 0.075*np.random.randn()
    return x,y


def test_reward(old, new, limbs_dict, joint_dict, **kwargs):
    """
     Dummy reward function used for testing.
    """
    prev_p1, _ = process_state(old, limbs_dict, joint_dict)
    next_p1, _ = process_state(new, limbs_dict, joint_dict)

    return 0.002*(next_p1['score'] - prev_p1['score'])


def adjusted_probability_reward(**kwargs):
    """
     Calculates the probability of the action 
     chosen by the lower level model based on the probability 
     distribution built using the higher level signal

     NOTE: See rewards_explained notebook for a visual
     on how this reward works
    """
    N = kwargs['number_actions']
    sig = kwargs['signal']
    action_idx = kwargs['action_idx']
    rv = stats.multivariate_normal(
        mean=sig[0:2], cov=[[sig[2], 0], [0, sig[3]]]
    )
    x,y = reverse_bin(action_idx, kwargs['controllable_actions'])
    return 4*(np.exp(rv.logpdf([x,y])) - 0.5)


def left_leg_reward(old, new, limbs_dict, joint_dict, **kwargs):
    return adjusted_probability_reward(**kwargs)

def right_leg_reward(old, new, limbs_dict, joint_dict, **kwargs):
    return adjusted_probability_reward(**kwargs)

def left_arm_reward(old, new, limbs_dict, joint_dict, **kwargs):
    return adjusted_probability_reward(**kwargs)

def right_arm_reward(old, new, limbs_dict, joint_dict, **kwargs):
    return adjusted_probability_reward(**kwargs)

def upper_body_reward(old, new, limbs_dict, joint_dict, **kwargs):
    return adjusted_probability_reward(**kwargs)

def lower_body_reward(old, new, limbs_dict, joint_dict, **kwargs):
    return adjusted_probability_reward(**kwargs)