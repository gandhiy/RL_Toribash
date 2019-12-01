# Deep Reinforcement Learning on Toribash with Expert Matches
Trained Policy                        |  Human Player                    
:------------------------:|:------------------------:
<img align="center" src=https://raw.githubusercontent.com/Miffyli/ToriLLE/master/images/toribash.gif alt="RL policy trained agent" width=250 height=250 /> | <img  align="center" src=https://git.cs.colorado.edu/yaga6341/csci-4831-7000/raw/master/images/replay.gif alt="Human agent" width=250 height=250 />

Content:

* [Toribash](##What-is-Toribash?)
* [Torille](##What-is-Torille?)
* [Getting Started](##Getting-Started)
* [Training](##Training-a-Model)


## What is Toribash?
[Toribash](https://www.toribash.com/) is a free to play two player game where each player controls a mannequin figure. The figure has 20 joints and 2 grips on each hand that can be changed between a number of states. The joints can all be in one of four positions, while the 2 grips will be either off or on. Each player decides which configuration to place their figure in and then the game moves forward a few frames. This process repeats for a set number of turns (with possibly varying frame numbers) and the game ends when there are no more frames left. The winner has the most points (deals the most injuries). 

## What is Torille?
A common library to train deep reinforcement learning (RL) models is openai's [gym](https://gym.openai.com/) and a forked library [stable-baselines](https://stable-baselines.readthedocs.io/en/master/). To make it easier to work with Toribash and these libraries, [Torille](https://github.com/Miffyli/ToriLLE) offers an environment wrapper that can send actions to a simplified game. By removing some of the shaders and visuals for playing the game, Torille offers a more feasible learning environment while running on hundreds of frames per second. It also offers the ability to parse saved matches from the Toribash community. 



## Getting Started

### Installing Requirements
Find the requirements file and run
``` bash
pip install -r requirements.txt
```
This will install all of the dependencies used in the project. 

### Downloading and Processing Matches

#### Expert Matches
Inside of [data](https://git.cs.colorado.edu/yaga6341/csci-4831-7000/blob/master/project/data) is a zip file of all of the replay matches. Unzip the files and place them inside of the replay folder under the torille project. On a Mac OS with, this can be found at /anaconda3/lib/python3.6/site-packages/torille/toribash/replay

To check if the replays are loaded correctly please run 
```bash
python test/test_replay.py
```

You should see a window pop up that first plays two agents doing nothing, and then plays a random replay sampled from the ones installed. It will also print out the number of current replays. To augment the training set, more replays can be added by going to the [Toribash Forum](https://forum.toribash.com/forumdisplay.php?f=10.)

Finally, to process all replays, run 
```bash
python load_all_replays.py
```

Now there should be two csv files representing information about player 1 and player 2 for all of the replays listed in the replays folder mentioned above. Visualizing this data can be difficult. The csv files can be many 10's of thousands of lines with around 150 columns for both player 1 and player 2, so there are a few examples in the [notebook folder](https://git.cs.colorado.edu/yaga6341/csci-4831-7000/tree/master/project/notebooks). 

#### Hidden Markov Model Training

#### Principle Component Analysis 

#### Linear Reward Weighting


## Training a Model

### Model Types
#### Vanilla
#### Multi-Limb
#### Hierarchy
### Configuration File
































## Exploring Data
Inside of the notebooks folder are 3 python notebooks that show some different interpreations of the data. The first, Exploring_Data, shows some visuals and elementary statistics about all of the expert matches. Second, Stochastic Similarity, provides a look into the stochastic similarity measure used in the v3 environments. Finally, Exploring_Action_Space shows what the actions chosen for the discrete action space look like both in dictionary form and how the actions look inside of the environment. 

## Training and Visualizing the Model
Inside of `model_trainer.py` there are various methods and options to change to train different models. The models and their relevant information are saved inside of the models folder and then can be viewed with the Model Runner notebook. 
