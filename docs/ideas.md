# Project Ideas
I would like to see if hierarchy RL with imitation learning can be used to learn complex control of [ToriLLE](https://github.com/Miffyli/ToriLLE). Here I describe some thoughts, ideas, and questions associated with this project. 

### Environment and Action Space
The agent is controlled through 20 different joints and has 2 different grips for both arms totalling to 4^21 possible combinations of actions. Each turn is a set of frames (for example 10 frames) where both agents change their joint positions and play through the frames. Score is based off damage to the other player. There is an underlying set of physics that is parameterized through a few different settings, but is not directly known to the player, therefore this can be considered a discrete action model-free reinforcement learning problem.

### Possible Tools to Use
* [ToriLLE](https://github.com/Miffyli/ToriLLE)
* [OpenAI Gym](https://gym.openai.com/)
* [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/guide/rl.html)
* [Tensorflow](https://www.tensorflow.org/)
* Visualization tools as needed

### Questions
* Can I limit the action space through a hierarchy?
* What kind of reward function will implicitly coordinate human-like behavior? Is human-like behavior a good result?
* Is a high scoring game, necessarily useful and can we learn anything useful from it?
* What do the visualizations look like?

### Possible Risks
* Not my code, so I'll have to learn about the environment code
* Hopefully, I have enough time to run all the experiments and gather all the results from those experiments
* I might not have as fine-grained control over the environment as I would like. 

