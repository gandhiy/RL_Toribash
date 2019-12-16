# Deep Reinforcement Learning on Toribash with Expert Matches
Trained Policy                        |  Human Player                    
:------------------------:|:------------------------:
<img align="center" src=images/policy_gif.gif alt="RL policy trained agent" width=250 height=250 /> | <img  align="center" src=images/replay.gif alt="Human agent" width=250 height=250 />

An example of some runs can be found [here](https://www.youtube.com/watch?v=9t0yV3qpQZY&feature=youtu.be)

Content:

* [Toribash](##what-is-toribash?)
* [Torille](##what-is-torille?)
* [Getting Started](##getting-started)
* [Training](##training-a-model)
* [Visualizing](##visualizing-a-model) 
* [Example](###example)
* [Experiments](##experiments)


## What is Toribash?
[Toribash](https://www.toribash.com/) is a free to play two player game where each player controls a mannequin figure. The figure has 20 joints and 2 grips on each hand that can be changed between a number of states. The joints can all be in one of four positions, while the 2 grips will be either off or on. Each player decides which configuration to place their figure in and then the game moves forward a few frames. This process repeats for a set number of turns (with possibly varying frame numbers) and the game ends when there are no more frames left. The winner has the most points (deals the most injuries). 

## What is Torille?
To make it easier to work with Toribash, [Torille](https://github.com/Miffyli/ToriLLE) offers an environment wrapper that can send actions to a simplified game. By removing some of the shaders and visuals for playing the game, Torille can run Toribash significantly faster (hundreds of frames per second) and can be used openai gym. Read the paper [here](https://arxiv.org/abs/1807.10110)!



## Getting Started

### Installing Requirements
Find the requirements file and run
``` bash
pip install -r requirements.txt
```
This will install all of the dependencies used in the project. 

### Downloading Matches
Inside of [data](data/) is a zip file of all of the replay matches. Unzip the files and place them inside of the replay folder under the torille project. On a Mac OS with, this can be found at /anaconda3/lib/python3.6/site-packages/torille/toribash/replay

To check if the replays are loaded correctly please run 
```bash
python test/test_replay.py
```

You should see a window pop up that first plays two agents doing nothing, and then plays a random replay sampled from the ones installed. It will also print out the number of current replays. To augment the training set, more replays can be added by going to the [Toribash Forum](https://forum.toribash.com/forumdisplay.php?f=10.)

Finally, to process all replays, run 
```bash
python load_all_replays.py
```

Now there should be two csv files representing information about player 1 and player 2 for all of the replays listed in the replays folder mentioned above. Visualizing this data can be difficult. The csv files can be many 10's of thousands of lines with around 150 columns for both player 1 and player 2, so there are a few examples in the [notebook folder](notebooks). 


## Training a Model
Training a model is relatively simple. Assign environment parameters within a yaml file and then run 

```bash
python model_trainer.py -c config.yml
```
Typical configuration files will look like this:
```yaml
env_dict: {
  agent: single, 
  timesteps: 50000,
  reward: 0,

  reward_kwargs: {
    'c0': 1.2,
    'c1': 1.2,
    'c2': 0.4,
},

  action_kwargs: {
    type: 'full',
  },

  savename: 'test_config'
}
``` 

A detailed explanation of the [configuration file](docs/configs.md) can be found in docs. 


## Visualizing a Model
Inside of the [notebook folder](https://git.cs.colorado.edu/yaga6341/csci-4831-7000/tree/master/project/notebooks/Model%20Runnder.ipynb), the Model Runner.ipynb will run a given model.

To run a model, there are two necessary files: the pickled model and a configuration file to load the environment. After training a model, the following folder structure will be available:

```
project
|-- models
|   |-- model1
|   |-- model2 (model just trained)
|   |   |-- logs
|   |   |   |-- monitor.csv
|   |   |-- model_0_of_5.pkl
|   |   |-- model_1_of_5.pkl
|   |   |-- .
|   |   |-- .
|   |   |-- .
|   |   |-- best_model.pkl
|   |   |-- final_model.pkl
|   |   |-- configs_dict.pkl
```


From here, choose a model to visualize and the configuration file (`configs_dict.pkl`). Change the appropriate parameters in the Model Runner.ipynb 
notebook and run the rest of the cells. Further instructions and explanations are 
provided in the notebook itself. 

### Example
**An example model and configs can be found in data. To use replace the following in the Model Runner.ipynb:**
```python
  model_name = None
  configs = "../models/{}/configs_dict.pkl".format(model_name)
  model = "../models/{}/best_model.pkl".format(model_name)
```
**with **
```python 
configs = "../data/example_dict.pkl"
model = "../data/example_model.pkl"
```



## Experiments

Running all of the models mentioned in the paper may take some time and, unfortunately, may also not produce exactly the same results. Given that, running 

```bash
python test/run_all.py
```

will generate all single agent policies. Other policies, like the multi-limb 
can be run by first generating models for each limb and then training the upper level controller (see [configs](docs/configs.md)).

To measure all of those models, run 
```bash
python test/metrics.py
```


















