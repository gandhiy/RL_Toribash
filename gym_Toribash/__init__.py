# register custom toribash environment
from gym.envs.registration import register


# Single Agent with access to full action space
register(
    id='Toribash-SingleAgentToribash-v0',
    entry_point='gym_Toribash.envs:SingleAgentToribash',
    kwargs={
        'reward_func': 0,
    }
)

register(
    id='Toribash-SingleAgentToribash-v1',
    entry_point='gym_Toribash.envs:SingleAgentToribash',
    kwargs={
        'reward_func': 1,
    }
)

register(
    id='Toribash-SingleAgentToribash-v2',
    entry_point='gym_Toribash.envs:SingleAgentToribash',
    kwargs={
        'reward_func': 2,
    }
)

register(
    id='Toribash-SingleAgentToribash-v3',
    entry_point='gym_Toribash.envs:SingleAgentToribash',
    kwargs={
        'reward_func': 3,
    }
)

# Individual limbs
register(
    id='Toribash-LeftLeg-v0',
    entry_point='gym_Toribash.envs:Left_Leg',
    kwargs={}
)

register(
    id='Toribash-RightLeg-v0',
    entry_point='gym_Toribash.envs:Right_Leg',
    kwargs={}
)

register(
    id='Toribash-LeftArm-v0',
    entry_point='gym_Toribash.envs:Left_Arm',
    kwargs={}
)

register(
    id='Toribash-RightArm-v0',
    entry_point='gym_Toribash.envs:Right_Arm',
    kwargs={}
)

register(
    id='Toribash-UpperBody-v0',
    entry_point='gym_Toribash.envs:Upper_Body',
    kwargs={}
)

register(
    id='Toribash-LowerBody-v0',
    entry_point='gym_Toribash.envs:Lower_Body',
    kwargs={}
)


# Multi Limb Models
register(
    id='Toribash-MultiLimb-v0',
    entry_point='gym_Toribash.envs:MultiLimbToribash',
    kwargs={
        'reward_func': 0
    }
)

register(
    id='Toribash-MultiLimb-v1',
    entry_point='gym_Toribash.envs:MultiLimbToribash',
    kwargs={
        'reward_func': 1
    }
)

register(
    id='Toribash-MultiLimb-v2',
    entry_point='gym_Toribash.envs:MultiLimbToribash',
    kwargs={
        'reward_func': 2
    }
)

register(
    id='Toribash-MultiLimb-v3',
    entry_point='gym_Toribash.envs:MultiLimbToribash',
    kwargs={
        'reward_func': 3
    }
)

# Hierarchy Model
# Majors
register(
    id='Toribash-MajorEnv-v0',
    entry_point='gym_Toribash.envs:MajorActions',
    kwargs={
        'reward_func': 0
    }
)

register(
    id='Toribash-MajorEnv-v1',
    entry_point='gym_Toribash.envs:MajorActions',
    kwargs={
        'reward_func': 1
    }
)

register(
    id='Toribash-MajorEnv-v2',
    entry_point='gym_Toribash.envs:MajorActions',
    kwargs={
        'reward_func': 2
    }
)


register(
    id='Toribash-MajorEnv-v3',
    entry_point='gym_Toribash.envs:MajorActions',
    kwargs={
        'reward_func': 3
    }
)


#Minors
register(
    id='Toribash-MinorEnv-v0',
    entry_point='gym_Toribash.envs:MinorActions',
    kwargs={
        'reward_func': 0
    }
)

register(
    id='Toribash-MinorEnv-v1',
    entry_point='gym_Toribash.envs:MinorActions',
    kwargs={
        'reward_func': 1
    }
)

register(
    id='Toribash-MinorEnv-v2',
    entry_point='gym_Toribash.envs:MinorActions',
    kwargs={
        'reward_func': 2
    }
)

register(
    id='Toribash-MinorEnv-v3',
    entry_point='gym_Toribash.envs:MinorActions',
    kwargs={
        'reward_func': 3
    }
)


#Details
register(
    id='Toribash-DetailEnv-v0',
    entry_point='gym_Toribash.envs:Details',
    kwargs={
        'reward_func': 0
    }
)

register(
    id='Toribash-DetailEnv-v1',
    entry_point='gym_Toribash.envs:Details',
    kwargs={
        'reward_func': 1
    }
)

register(
    id='Toribash-DetailEnv-v2',
    entry_point='gym_Toribash.envs:Details',
    kwargs={
        'reward_func': 2
    }
)

register(
    id='Toribash-DetailEnv-v3',
    entry_point='gym_Toribash.envs:Details',
    kwargs={
        'reward_func': 3
    }
)
