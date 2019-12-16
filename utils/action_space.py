"""
 Build different models to generate action manuevers. Essentially, instead of having the network learn over 
 22 outputs, learn a single output that leads to a prespecified action.

"""
import numpy as np

from time import sleep
from copy import deepcopy
from scipy import stats
from collections import Counter
from utils.tools import *
from utils.viz_tools import visuals
from torille import ToribashControl, constants


class action_space:
    def __init__(self, csv_path):
        """
         Build a class of methods to interpret actions from. Actions built from expert matches
         Most classes will be structure as such:

         action_space.builder(number_actions, **kwargs)
         
         This way, the number of actions can be a parameter to the learning model. Use with SingleAgentDiscreteAction
         agent. Every method should return a dictionary {0: action0, 1:action1, ... N:actionN}

         Args:
            csv_path (filepath): path to the loaded replays
        """
        self.df = load_csv(csv_path)
        self.vis = visuals(csv_path)
        self.vis.actions = True
        
    def frequency_builder(self, number_actions, controllable=constants.NUM_CONTROLLABLES, **kwargs):
        """
         Return the N most frequent actions 
        """
        ret = {}
        actions = np.array(self.df[self.vis.actions])
        if(type(controllable) is list):
            actions = actions[:, controllable]
        # this lets us separate the unique actions from their repeated calls
        u, idx = np.unique(actions, axis=0, return_inverse=True)
        # build a counter on the indices
        c = Counter(idx)
        # build the dictionary to return
        counter = 0
        for i, count in c.most_common(number_actions):
            ret[counter] = list(u[i].astype(int))
            counter += 1
        return ret

    def temporal_frequency_builder(self, number_actions, controllable=constants.NUM_CONTROLLABLES, **kwargs):
        """
         Return the N most frequent actions weighted by how close to the 
         half way-point of the game they were played at (i.e. a gaussian centered 
         at number_of_turns/2)
        """
        ret = {}
        # begin similarly to frequency builder
        actions = np.array(self.df[self.vis.actions])
        if(type(controllable) is list):
            actions = actions[:, controllable]

        _,idx = np.unique(actions, axis=0, return_inverse=True)
        c = Counter(idx)

        # we're going to need a temporary dataframe to add columns to
        tmp = deepcopy(self.df)

        # get the frequency count for all actions from the dataframe
        list_of_freq = []
        for i in idx:
            list_of_freq.append(c[i])
        # add normalized frequencies to the dataframe
        tmp['norm_freq'] = np.array(list_of_freq)/len(tmp)
        
        # now calculate the weighted frequencies
        new_freq = []
        for match in self.vis.matches:
            tmp_match = tmp[tmp['match'] == match]
            num_turns = max(tmp_match['turn'])
            # build a gaussian for each match
            rv = stats.norm(loc=num_turns/2., scale=num_turns/2.)
            # multiply the norm frequencies with the weights
            new_freq += list(tmp_match['norm_freq']*rv.pdf(tmp_match['turn']))        
        tmp['weighted_freq'] = new_freq
        # sort by the weighted frequencies
        sorted_df = tmp.sort_values('weighted_freq', ascending=True)
        # remove duplicate actions again
        unique_sorted_df = sorted_df[self.vis.actions].drop_duplicates(subset=self.vis.actions)
        new_actions = np.array(unique_sorted_df)
        if(type(controllable) is list):
            new_actions = new_actions[:, controllable]

        for i in range(number_actions):
            ret[i] = list(new_actions[i].astype(int))
        return ret
        
    def random_actions(self, number_actions, controllable=constants.NUM_CONTROLLABLES, **kwargs):
        """
         Return N random actions
        """
        ret = {}
        if(type(controllable) is list):
            l = len(controllable)
        else:
            l = 22
        for i in range(number_actions):
            ret[i] = list(np.random.randint(1,5,size=(l,)))
        return ret

def bin_val(x,y, T, k=21):
    """
     Defines the grid for continuous actions on single agents

     Args:
        x (float): the x value outputted by the model
        y (float): the y value outputted by the model
        T (int): the hieght and width of the grid (assumes square grid)

     Returns:
        bin (int): the bin number that the output falls into. 
    """

    b_size = T/np.sqrt(4**k)
    bx = (x//b_size)
    by = (y//b_size)
    return np.sqrt(4**k)*by + bx

def watch_actions(actions, pause=1):
    """
     Watch the set of actions chosen by builder. Note, this does not play the action 
     within any temporal context, so it could be a decent action but hard to visualize as 
     the first action. 

     Args:
        actions (dict): a dictionary for the actions. See action_space for methods to produce these
        actions
        pause (float): how long to pause the action
    """

    try:    
        controller = ToribashControl(draw_game=True)
        controller.init()
        _, t = controller.get_state()
        for i in actions.keys():
            act = actions[i]
            action = [act, [3]*constants.NUM_CONTROLLABLES]
            action[1][-2] = 1
            action[1][-1] = 1
            controller.make_actions(action)
            _, t = controller.get_state()
            sleep(pause)
            controller.finish_game()
            if(controller.requires_reset):
                _ = controller.reset()
        controller.close()
    except RuntimeWarning as inst:
        print("manual controller shutdown")
        controller.close()
        print("error: {}".format(inst))