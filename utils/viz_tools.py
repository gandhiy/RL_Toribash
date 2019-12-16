# A set of useful tools for visualizations. 
# Also, some handy functions for getting access 
# to subsections of the dataframe (i.e. joint 
# velocities, actions, etc)

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from torille import ToribashControl
from utils.tools import load_csv, create_dictionary_mapper


class visuals:
    def __init__(self, csv_path):
        """
         A tool set for building different visualizations from expert matches
         as well as accessing information about the dataframe such as getting specific
         groupings of columns that would be useful in this context: joints, actions, limbs. 


         Args:
            csv_path (filepath): path to the loaded replays
        """


        self.df = load_csv(csv_path)
        self.matches = self.df['match'].unique()
        self.matchName = self.matches[0]
        self.match_df = self.df[self.df['match'] == self.matchName]
        self.grips = False

        # build dictionaries for joints and limbs
        self.dbp = create_dictionary_mapper()
        self.djp = create_dictionary_mapper(False)

    @property
    def match(self):
        """
         Returns the current matchName and its index in the matches
        """
        return self.matchName, np.where(self.matches == self.matchName)[0][0]
    
    @match.setter
    def match(self, matchIdx):
        """
         Set the current match using the name of the match or an index value. 

         Args:
            matchIdx (string || int): Match to use for visualizations
        """
        if(type(matchIdx) is int):
            if(matchIdx >= len(self.matches)):
                raise ValueError("matchIdx: {} must be less than {}".format(matchIdx, len(self.matches)))
            else:
                self.matchName = self.matches[matchIdx]
        elif(type(matchIdx) is str): 
            if(matchIdx not in self.matches):
                raise ValueError("{} is not a valid match name".format(matchIdx))
            else:
                self.matchName = matchIdx
        else:
            raise ValueError("matchIdx must be either an int or string")

        self.match_df = self.df[self.df['match'] == self.matchName]


    @property
    def positions(self):
        return [c for c in self.df.columns if "pos" in c]
    
    @property
    def velocities(self):
        return [c for c in self.df.columns if "vel" in c]


    @property
    def actions(self):
        cols =  [c for c in self.df.columns if 'act_' in c]
        if(not self.grips):
            for el in ['act_left_grip', 'act_right_grip']:
                cols.remove(el)
        return cols
    
    @actions.setter
    def actions(self, grips):
        """
         Setting actions to True or False 
         adds or removes the grips from 
         the actions list respectively. 
        """
        self.grips = grips

    
    def plot_column(self, colName):
        """
         Plots a pd.DataFrame column for each turn of the currently set match

         Args:
            colName (string): column to plot
         Return:
            plot (matplotlib.pyplot.axes): matplotlib plot axes. This allows for 
            other information to be added. 
        """
        if(colName == 'score'):
            alpha = 10/max(self.match_df[colName])
            return plt.plot(alpha*self.match_df[colName], '.')
        return plt.plot(self.match_df[colName], '.')    

    def plot_action_hist(self, colName):
        """
         Plots the histogram (normalized) of a action column
        """
        
        if(colName not in self.actions):
            raise ValueError("This method is only for action columns")

        ts = self.match_df[colName]
        to_plot = ts.value_counts().sort_index()/len(ts)
        return to_plot.plot(kind='bar')

    def heatmap(self):
        """
         Plot the heatmap of the current match with the 
         currently set actions. 

         To include the grip actions in the viz.actions 
         list, make sure to set viz.actions = True beforehand
        """
        ts = self.match_df[self.actions]
        return sns.heatmap(ts.transpose(), cbar_kws=dict(ticks=[0,1,2,3,4]), cmap='viridis')

    def watch_replay(self):
        """
         Plays the currently selected match in a Toribash game window
        """
        print("Playing Match: {}".format(self.match))

        try:
            controller = ToribashControl(draw_game=True)
            controller.init()
            controller.finish_game()
            states = controller.read_replay(self.match[0])
            controller.close()
        except RuntimeError as inst:
            print("manual controller shutdown")
            controller.close()
            print("error: {}".format(inst))