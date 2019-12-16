import os
import pandas as pd


def _fix_grips(df):

    # The joints and the grips have different domains. 
    # The joints are in {1, 2, 3, 4} whereas the grips on the agent
    # are in {0,1}. For the dataframe, it is easier to interpret and 
    # analyze using the same domain {1,2,3,4} where grips can only be 
    # 1 or 2. 
    df['act_left_grip'] +=  1
    df['act_right_grip'] += 1
    return df

def create_dictionary_mapper(limbs=True):
    """
     Process a text file into a dictionary
     Args:
        limbs (bool): if true, returns a dictionary of limbs components, else
        returns a dictionary of controllable joints
    Return:
        ret (dict): the dictionary specified by limbs
    
    """
    data_path = os.path.join(os.path.dirname(__file__), '../data/')
    # using explicit paths because of some weird errors when loading environment in notebook
    if(limbs):
        path = os.path.join(data_path, 'body_parts.txt')
    else:
        path = os.path.join(data_path, 'joint_parts.txt')
    ret = {}
    with open(path, 'r') as f:
        for line in f:
            ret[int(line.strip().split(":")[0])] = line.strip().split(":")[1]
    return ret


def process_vectors(v, m):
    """
     A helper function to map position and velocity vectors to x,y, and z
     keys of a dictionary.

    Args:
        v (dict): The original dictionary from load_replays
        m (dict): A matching dictionary of body part or joint names

    """
    ret = {}
    for k in m.keys():
        # if v.shape -> (?,3) then v represents body part position or velocity 
        if(len(v.shape) > 1 and v.shape[1] == 3):
            for i in range(0,3):
                # add x,y,z values from position and velocity of each body part to the dictionary
                ret[m[k].replace(' ', '_') + '_' +chr(ord('x') + i)] = v[k][i]
        else: # if v.shape -> (?,) then v represents actions/joint configurations
            ret[m[k].replace(' ', '_')] = v[k]
    return(ret)

def load_csv(filepath):
    """
     Recommended for loading a pandas dataframe and using with utils.viz_tools

     Args:
        filepath (string): The path to the csv for player information. 
        If no such file created yet, run load_replay. 
    """
    # if the csv was already saved,
    # this will return a pandas dataframe
    return _fix_grips(pd.read_csv(filepath))


def flatten_dict(p):
    """
     Flatten a dictionary that is a mix of dictionaries and single values

     Args:
         p (dict): a dictionary of vectors and dictionaries (maximum depth 
         allowed = 2) to flatten
    """
    f = {}
    for k in p.keys():
        if(type(p[k]) is dict):
            # iterate over the sub-keys in any dictionary within p
            for k_sub,v in p[k].items():
                f[k+"_"+k_sub] = v
        else: # if p[k] isn't a dictionary, just set f[k] to p[k]
            f[k] = p[k]
    return f

def process_state(state, limbs, joints):
    """
     Process the ToribashState into a dictionary for easy
     access of all of the variables.

     NOTE: injuries represents the score of the other player! Player 1's score 
     is held within player 2's state information.
    """
    p1_row = flatten_dict({
            'pos': process_vectors(state.limb_positions[0], limbs), 
            'vel': process_vectors(state.limb_velocities[0], limbs), 
            'act': process_vectors(state.joint_states[0], joints), 
            'score': state.injuries[1]}
        )
    # player 2 information
    p2_row = flatten_dict({ 
            'pos': process_vectors(state.limb_positions[1], limbs), 
            'vel': process_vectors(state.limb_velocities[1], limbs), 
            'act': process_vectors(state.joint_states[1], joints), 
            'score': state.injuries[0]}
        )
    return p1_row, p2_row



if __name__ == "__main__":
    print(create_dictionary_mapper(True))
