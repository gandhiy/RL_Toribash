"""
Reward functions for training agents.
"""


import numpy as np
import pandas as pd


from utils.tools import process_state, load_csv


def _get_information(old, new, limbs, joints):
    # processes input states and returns dictionaries
    prev_p1, prev_p2 = process_state(old, limbs, joints)
    next_p1, next_p2 = process_state(new, limbs, joints)
    return prev_p1, prev_p2, next_p1, next_p2

def _safe_log(x):
    #safe log to normalize large in-game scores
    if x < 0:
        return -np.log10(abs(x))
    elif x == 0:
        return -1
    else:
        return np.log10(x)
   

def _check_win(score1, score2, terminal):
    # who's winning
    if not terminal:
        if(score1 > score2):
            return 1
        else: 
            return -0.1
    else:
        if(score1 > score2):
            return 1
        else: 
            return -1

def _get_com_dist(prev_p1, prev_p2):
    # calculate difference in center of motion. assume uniform weighting
    x_keys = ['pos_breast_x','pos_chest_x','pos_groin_x','pos_head_x','pos_l_biceps_x','pos_l_butt_x','pos_l_foot_x','pos_l_hand_x','pos_l_leg_x','pos_l_pecs_x','pos_l_thigh_x','pos_l_triceps_x','pos_r_biceps_x','pos_r_butt_x','pos_r_foot_x','pos_r_hand_x','pos_r_leg_x','pos_r_pecs_x','pos_r_thigh_x','pos_r_triceps_x','pos_stomach_x']
    y_keys = ['pos_breast_y','pos_chest_y','pos_groin_y','pos_head_y','pos_l_biceps_y','pos_l_butt_y','pos_l_foot_y','pos_l_hand_y','pos_l_leg_y','pos_l_pecs_y','pos_l_thigh_y','pos_l_triceps_y','pos_r_biceps_y','pos_r_butt_y','pos_r_foot_y','pos_r_hand_y','pos_r_leg_y','pos_r_pecs_y','pos_r_thigh_y','pos_r_triceps_y','pos_stomach_y']
    z_keys = ['pos_breast_z','pos_chest_z','pos_groin_z','pos_head_z','pos_l_biceps_z','pos_l_butt_z','pos_l_foot_z','pos_l_hand_z','pos_l_leg_z','pos_l_pecs_z','pos_l_thigh_z','pos_l_triceps_z','pos_r_biceps_z','pos_r_butt_z','pos_r_foot_z','pos_r_hand_z','pos_r_leg_z','pos_r_pecs_z','pos_r_thigh_z','pos_r_triceps_z','pos_stomach_z']
    x1_com = np.average([v for k,v in prev_p1.items() if k in x_keys])
    y1_com = np.average([v for k,v in prev_p1.items() if k in y_keys])
    z1_com = np.average([v for k,v in prev_p1.items() if k in z_keys])
    x2_com = np.average([v for k,v in prev_p2.items() if k in x_keys])
    y2_com = np.average([v for k,v in prev_p2.items() if k in y_keys])
    z2_com = np.average([v for k,v in prev_p2.items() if k in z_keys])

    p1 = np.array([x1_com, y1_com, z1_com]).T
    p2 = np.array([x2_com, y2_com, z2_com]).T

    return np.linalg.norm(p1 - p2)


def _score_only_reward(old, new, limbs, joints, **kwargs):
    # score based reward
    prev_p1, prev_p2, next_p1, next_p2 = _get_information(old,new,limbs,joints)

    player_1_gain = next_p1['score'] - prev_p1['score']
    player_2_gain = next_p2['score'] - prev_p2['score']
    return kwargs['c0']*np.sign(player_1_gain - player_2_gain) - kwargs['c1']*_safe_log(prev_p1['score'] + kwargs['c1']*_safe_log(next_p1['score']))


def transition_constrained_reward(old, new, limbs, joints, **kwargs):
    """
    Reward 0: 
    Reward is based on score of both players and also constrains the agent's actions.
    """
    prev_p1, _, next_p1, _ = _get_information(old,new,limbs,joints)

    score_reward = _score_only_reward(old, new, limbs, joints, **kwargs)
    prev_acts = np.array([prev_p1[k] for k in prev_p1.keys() if 'act' in k])
    next_acts = np.array([next_p1[k] for k in next_p1.keys() if 'act' in k])
    a_pen = np.linalg.norm(prev_acts - next_acts, ord=1)

    if(a_pen < 1):
        return score_reward - 100
    elif(a_pen > 11): # 11 = 22/2
        return score_reward - 100
    else:
        return score_reward - (kwargs['c2'] * a_pen)


def linreg_reward(old, new, limbs, joints, **kwargs):
    """
    Reward 1:
    Use a set of weights learned from the expert matches 
    to calculate reward
    """
    phi = kwargs['phi']
    prev_p1, prev_p2, next_p1, next_p2 = _get_information(old,new,limbs,joints)


    prev_p1 = np.array([v for k,v in prev_p1.items() if k is not 'score'], dtype=np.float)
    prev_p2 = np.array([v for k,v in prev_p2.items() if k is not 'score'], dtype=np.float)
    next_p1 = np.array([v for k,v in next_p1.items() if k is not 'score'], dtype=np.float)
    next_p2 = np.array([v for k,v in next_p2.items() if k is not 'score'], dtype=np.float)
    prev_vals = np.concatenate((prev_p1,prev_p2), axis=0)
    next_vals = np.concatenate((next_p1, next_p2), axis=0)
    reward = kwargs['c0'] * phi.dot(prev_vals)
    reward += kwargs['c1'] * phi.dot(next_vals)    
    return reward 

def dist_reward(old, new, limbs, joints, **kwargs):
    """
    Reward 2:
    Return the score of the game minus the distance from the opponent
    """
    prev_p1, prev_p2, _, _ = _get_information(old, new, limbs, joints)

    dist = _get_com_dist(prev_p1, prev_p2)
    return (kwargs['c0']*_safe_log(prev_p1['score'])) - kwargs['c1']*dist


 
def curriculum_reward(old, new, limbs, joints, **kwargs):
    """
    Reward 3:
    Curriculum reward (based on percent of training done).
        0 < t < 0.25:
            just win
        0.25 < t < 0.5:
            win and score more points
        0.5 < t < 0.75:
            win, score more points, and stay close to the opponent
        0.5 < t < 0.75:
            win, score more points, stay close to the opponent, and act like the experts
    """

    percent_done = kwargs['iteration']/kwargs['timesteps']
    prev_p1, prev_p2, next_p1, next_p2 = _get_information(old,new,limbs,joints)
    p1_score = next_p1['score']
    p2_score = next_p2['score']
    
    penalty = prev_p1['score'] - next_p1['score']

    if(percent_done < 0.25):
        return _check_win(p1_score, p2_score,kwargs['terminal'])
    elif(percent_done >= 0.25 and percent_done < 0.5):
        return _check_win(p1_score, p2_score, kwargs['terminal']) - kwargs['c2']*_safe_log(penalty) 
    elif(percent_done >= 0.5 and percent_done < 0.75):
        return _check_win(p1_score, p2_score, kwargs['terminal']) - kwargs['c2']*_safe_log(penalty) - kwargs['c3']*_get_com_dist(prev_p1, prev_p2)
    elif(percent_done >= 0.75):
        return _check_win(p1_score, p2_score, kwargs['terminal']) - kwargs['c2']*_safe_log(penalty) - kwargs['c3']*_get_com_dist(prev_p1, prev_p2) + kwargs['c4']*linreg_reward(old, new, limbs, joints, **kwargs)
    
