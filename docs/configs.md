# Configuration File Explanations

* [Agents](##Agents)
* [Actions](##Actions)
* [Rewards](##Rewards)

## Agents

key: `agent`

options: `single`, `multi`, `limb`, `hierarchy`

`single`: This loads in the vanilla single agent reinforcement learning environment. The action space, here, is the entire figure and whatever action type is chosen will control all of the joints with a single model. 

`multi`: This loads in an environment to train a high level controller for 6 different models that each control a section of the toribash agent. This model works similar to the `single` agent model, but it learns, as its action, to send a signal ( a vector representing a 2D distribution), that the lower limb models have been trained to interpret. To use this environment, you must specify a trained model for each section as shown below and add it to the configuration file. 

```yaml

  limb_models: {
    left_leg_model: left_leg.pkl,
    right_leg_model: right_leg.pkl,
    left_arm_model: left_arm.pkl,
    right_arm_model: right_arm.pkl,
    upper_body_model: upper_body.pkl,
    lower_body_model: lower_body.pkl
  }

```

`limb`: This trains a specific section of the toribash agent for the `multi` agent type. Each model's reward is how well they interpret a randomly sent signal. To train one of these models, the configuration file also needs the name of the environment. 

```yaml
limb: LeftLeg
```
The correct limb names (and which joints they control) are:
* LeftLeg (left hip, left knee, left ankle)
* RightLeg (right hip, right knee, right ankle)
* LeftArm (left shoulder, left elbow, left wrist, left grip)
* RightArm (right shoulder, right elbow, right wrist, right grip)
* UpperBody (neck, chest, right pec, left pec)
* LowerBody (lumbar, abs, right glute, left glute)

`hierarchy`: The final model is similar to the `multi` type, because it uses multiple models of control, but this agent type embeds those models within one another rather than using one a high level controller. The three environments, major, minor, and detailed, build upon the previous level (minor builds on major and detailed builds on minor). To use this model type, the level you wish to train needs to be specified. 

```yaml
level: [major, minor, detail]
```

Also, depending on which level is chosen, the the configuration file needs to point to other trained models in the following manner:

```yaml
inner_models: {
    major_model: 'major_model.pkl',
    major_model_configs: 'major_model_configs.pkl',
    minor_model: 'minor_model.pkl',
    minor_model_configs: 'minor_model_configs.pkl'
}
```
When training the major model, `inner_model` does not need to be defined. When training the minor model, `major_model` and `major_model_configs` must be defined. And when training the detail model, all of the settings above must be defined. 

## Actions
For all of the single agent policies and the embedded policy, the `action_kwargs`
should be specified. 

* Multi-Discrete 

Full discrete action space.
```yaml
action_kwargs: {
    type: 'full',
  }
```

* Random

N random actions sampled from the expert matches.
```yaml
action_kwargs: {
    type: 'discrete',
    method: 'random',
    num_actions: 30
  }
```

* Frequent

N most frequent actions from the expert matches.
```yaml
action_kwargs: {
    type: 'discrete',
    method: 'frequent',
    num_actions: 30
  }
```

* Weighted

N most frequent actions weighted by a gaussian kernel across the match.
```yaml
action_kwargs: {
    type: 'discrete',
    method: 'weighted',
    num_actions: 30
  }
```

* Continuous

Model outputs coordinates within the low-high range given and the model interprets that as falling within a specific action bin. 
```yaml
action_kwargs: {
    type: 'continuous',
    low: 0,
    high: 15
  }
```


## Rewards
There are a few different rewards that work with each of the above models. For a detail look into each reward function, please refer to [`rewards.py`](../project/utils/rewards.py). Below we give a short description of the reward and some suggested arguments. If the reward_weights are not available, please run 
```bash
python utils/train_Ridge.py
```

key: `reward`

options: `0`, `1`, `2`, `3`

`0`: Reward 0 is the score only reward. This measures the agent's ability at every time step based solely on the in-game score. 

```yaml
reward_kwargs: {
  'c0': 1.2,
  'c1': 1.2,
  'c2': 0.4,
}
```


`1`: Reward 1 uses an inverse reward design type formulation. We use an identity function on the state values though. 

```yaml
reward_kwargs: {
  'c0': 1.3,
  'c1': 0.9,
  'phi': 'data/reward_weights.pkl'
}
```

`2`: Reward 2 measures the distance from the center of motion of both Toribash figures. The farther from the opponent, the greater the negative penalty. 

```yaml
reward_kwargs: {
  'c0': 0.2,
  'c1': 1.9,
}
```

`3`: Reward 3 uses a curriculum to help the agent learn control over the environment. 

```yaml
reward_kwargs: {
  'c0': 0.5,
  'c1': 0.5,
  'c2': 1.0,
  'c3': 5.0,
  'c4': 0.25,
  'phi': 'data/reward_weights.pkl'
}
```