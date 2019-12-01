import numpy as np
import pandas as pd


from utils.tools import process_state, load_csv


def get_information(old, new, limbs, joints):
    # reward function idx: 0
    prev_p1, prev_p2 = process_state(old, limbs, joints)
    next_p1, next_p2 = process_state(new, limbs, joints)
    return prev_p1, prev_p2, next_p1, next_p2


def _check_win(score1, score2, terminal):
    if not terminal:
        if(score1 > score2):
            return 1
        else: 
            return -0.1
    else:
        if(score1 > score2):
            return 100
        else: 
            return -100


def score_only_reward(old, new, limbs, joints, **kwargs):
    prev_p1, prev_p2, next_p1, next_p2 = get_information(old,new,limbs,joints)

    player_1_gain = next_p1['score'] - prev_p1['score']
    player_2_gain = next_p2['score'] - prev_p2['score']
    return np.sign(player_1_gain - player_2_gain)*kwargs['c0']*(next_p1['score'] - prev_p1['score'])


def transition_constrained_reward(old, new, limbs, joints, **kwargs):
    prev_p1, prev_p2, next_p1, next_p2 = get_information(old,new,limbs,joints)

    score_reward = score_only_reward(old, new, limbs, joints, **kwargs)
    prev_acts = np.array([prev_p1[k] for k in prev_p1.keys() if 'act' in k])
    next_acts = np.array([next_p1[k] for k in next_p1.keys() if 'act' in k])
    return score_reward - (kwargs['c1'] * np.linalg.norm(prev_acts - next_acts, ord=1))


def linreg_reward(old, new, limbs, joints, **kwargs):
    phi = kwargs['phi']
    prev_p1, prev_p2, next_p1, next_p2 = get_information(old,new,limbs,joints)

    pp1 = prev_p1.pop('score')
    pp2 = prev_p2.pop('score')
    np1 = next_p1.pop('score')
    np2 = next_p2.pop('score')
    prev_p1 = np.array([v for k,v in prev_p1.items() if k is not 'score'], dtype=np.float32)
    prev_p2 = np.array([v for k,v in prev_p2.items() if k is not 'score'], dtype=np.float32)
    next_p1 = np.array([v for k,v in next_p1.items() if k is not 'score'], dtype=np.float32)
    next_p2 = np.array([v for k,v in next_p2.items() if k is not 'score'], dtype=np.float32)
    prev_vals = np.concatenate((prev_p1,prev_p2), axis=0)
    next_vals = np.concatenate((next_p1, next_p2), axis=0)
    reward = phi.dot(prev_vals)
    reward += phi.dot(next_vals)    
    return reward + np.sign(np1 - pp1 - np2 + pp2)*kwargs['c0']*(np1 - pp1)


def curriculum_reward(old, new, limbs, joints, **kwargs):
    """
     Based on the percent of the total timesteps that have occured change the reward function.
     
    """

    percent_done = kwargs['iteration']/kwargs['timesteps']
    _, _, next_p1, next_p2 = get_information(old,new,limbs,joints)
    p1_score = kwargs['c0']*next_p1['score']
    p2_score = kwargs['c0']*next_p2['score']
    

    #TODO: adjust numbers to be reasonable
    if(percent_done < 0.25):
        return _check_win(p1_score, p2_score,kwargs['terminal'])
    elif(percent_done >= 0.25 and percent_done < 0.5):
        return _check_win(p1_score, p2_score, kwargs['terminal']) + (p1_score/(1 + p2_score + percent_done*p2_score))
    elif(percent_done >= 0.5):
        return _check_win(p1_score, p2_score, kwargs['terminal']) + kwargs['c2']*linreg_reward(old, new, limbs, joints, **kwargs)
    
