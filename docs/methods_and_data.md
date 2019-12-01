## Methods and Data Prep

### Data
The data consist of frame by frame match state vectors. These are from human-played replays where each state vector is approximately 150 different features. 
These features are a part of a few different categories of game observations:

* 22 actions joints and grips
* (x,y,z) positions of the 21 different body parts
* (x,y,z) velocities of the 21 different body parts
* match statistics including turn number, replay name, and score

With just the initial replays pre-downloaded with the game, there are 44729 different observations for 
player one and player two each. While player one is always active during the game, sometimes player 2 is in dummy mode (default action every turn). 

### Prep
The data will be loaded into a pandas dataframe. After some initial testing, it takes about 15 to 20 minutes to load in the 41 replays, so a few more may be added and a few may be switched out with other replays that are more interesting samples (longer matches, more complex movements, etc). To minimize loading time, I'll save the match values into a csv and read that in rather than play through all of the replays each time. 

### Data Visualization
Visualizing all of this data with one graphic is just as much of a challenge as making an algorithm for learning a RL policy. To represent how an agent moves through a space, I may aggregate all of the different position and velocity vectors into a center of mass vector and plot that. Plotting the actions in a heatmap vs time (turns), will also provide a more visual understanding of how the actions change over time. 

![heatmap](https://git.cs.colorado.edu/yaga6341/csci-4831-7000/raw/master/images/example_heatmap.jpg)



### Methods
The expert replays will filter into two different aspects of this project. First, they will server as observations for behavioral cloning algorithms. Second, the will go through a wide variety of clustering algorithms to try and establish a more rigorous human baseline. The behavioral cloning can then be used as a primer for PPO or TRPO - two very commonly used deep RL methods. Once, the RL algorithm has been sufficiently trained, the generated trajectories will be measured against the expert matches and qualitatively and quantitatively measured for their "humanness". 


