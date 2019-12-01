import pickle
import numpy as np
import pandas as pd 

from os.path import join
from utils.tools import load_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def train_ridge(player1_csv, player2_csv, save_to):
    # load in the data
    df1 = load_csv(player1_csv)
    df2 = load_csv(player2_csv)
   

    # We'll be joining both players dataframes, so let's 
    # add some identifiers for the 2nd player
    rename_func = lambda x: x+'_player_2'
    df2_renamed = df2.rename(columns=rename_func)
    df1[df2_renamed.columns] = df2_renamed

    # regress non-zero reward transitions
    df1['score_t_next'] = df1['score'].shift()
    df1 = df1.dropna()    
    df1 = df1.drop(index = df1[df1['score_t_next'] - df1['score'] == 0].index)
    y = df1.score/max(df1.score)

    # drop some unused columns
    df = df1.drop(columns=[
    'score_t_next',
    'turn', 
    'turn_player_2',
    'Unnamed: 0', 
    'Unnamed: 0_player_2', 
    'match_player_2', 
    'match', 
    'turn_player_2',
    'score',
    'score_player_2'])
    
    # create the X matrix and split the train and test data
    X = np.array(df)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

    # deliver some details about the method run to the 
    # user
    clf = Ridge(alpha=6.5, max_iter=1000)
    clf.fit(X_train,y_train)
    print("Score: ", clf.score(X_test, y_test), "with alpha: {}".format(6.5))

    # Save the weights, they'll be used later for testing the trajectories.
    with open(save_to+'_reward_weights.pkl', 'wb') as f:
        pickle.dump(clf.coef_, f)
    
    return clf.coef_
