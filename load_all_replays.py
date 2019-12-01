# This file only needs to be called once during the beginning if
# behavior control methods are being used. 
# 
# Installing torille through pypi will not install replays. General 
# replays can be installed either with Toribash (https://www.toribash.com/) 
# or specific installs of replays can be done through the toribash forum 
# (https://forum.toribash.com/forumdisplay.php?f=10). To download from the 
# forum, it is necessary to create an account.


import os 
import pandas as pd

from utils.tools import *
from torille import ToribashControl



def _load_replay(REPLAY_FILE):
    """
     Loading single replay
    """

    # Closes the controller even if there is a error somewhere in the state recording process
    try:
        controller = ToribashControl(draw_game=False) # create controller
        controller.init() # init
        controller.finish_game() # can't start controller with a replay, has to load it in after at least one game
        states = controller.read_replay(REPLAY_FILE) # play and record replay file
        controller.close() # close controller
        return states 
    except RuntimeError as err:
        print("There was a runtime error causing the program to stop and potentially not close the game window")
        print("Closing game window")
        controller.close()
        print(err)


def load_replay(verbose=False):
    """ 
     Load all replays in replay folder from Torille 
     into a csv file with all state information.

     Args:
         verbose (bool): prints the current replay being loaded
    """

    # create dictionaries from body parts and joints text files
    limbs = create_dictionary_mapper(True)
    joints = create_dictionary_mapper(False)

    # location of replay files to load into csv
    # add replays here to add to match collections
    replay_folder = '/anaconda3/lib/python3.6/site-packages/torille/toribash/replay'
    replays = [f for f in os.listdir(replay_folder) if '.rpl' in f]

    P1 = []
    P2 = []
    for i,replay in enumerate(replays):
        if(verbose):
            print("Loading replay {}. Replay number {} of {}".format(replay, i+1, len(replays)))
        states = _load_replay(replay)
        
        for i,state in enumerate(states):
            # process state information into dictionary for player 1 and player 2 
            p1_row, p2_row = process_state(state, limbs, joints)
            p1_row['match'] = replay
            p1_row['turn'] = i
            p2_row['match'] = replay
            p2_row['turn'] = i
            
            P1.append(p1_row)
            P2.append(p2_row)

    df1 = _fix_grips(pd.DataFrame(P1))
    df2 = _fix_grips(pd.DataFrame(P2))

    print("Saving csv's for player 1 and 2 to data folder")
    os.makedirs('data', exist_ok=True)
    df1.to_csv("data/player1_state_info.csv")
    df2.to_csv("data/player2_state_info.csv")
    return (df1, df2)



if __name__ == "__main__":
    load_replay(verbose=True)