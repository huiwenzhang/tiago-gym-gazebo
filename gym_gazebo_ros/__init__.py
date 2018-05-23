from gym.envs.registration import register

register(
    id='Assembler-v0',
    entry_point='gym_gazebo_ros.envs.assembler_robot:AssemblerEnv',
    max_episode_steps=1000,
    reward_threshold=4.1,
)

register(
    id='Assembler-v1',
    entry_point='gym_gazebo_ros.envs.assembler_robot:AssemblerPiHEnv',
    max_episode_steps=1000,
    reward_threshold=4.1,
)

register(
    id='Assembler-v2',
    entry_point='gym_gazebo_ros.envs.assembler_robot:AssemblerPiHv2Env',
    max_episode_steps=1000,
    reward_threshold=4.1,
)

# alvin for tiago
register(
    id='Tiago-v0',
    entry_point='gym_gazebo_ros.envs.tiago_robot:TiagoEnv',
    max_episode_steps=1000,
    reward_threshold=4.1,
)

register(
    id='TiagoReach-v0',
    entry_point='gym_gazebo_ros.envs.tiago_robot:TiagoReachEnv',
    max_episode_steps=1000,
    reward_threshold=4.1,
)

register(
    id='TiagoPick-v0',
    entry_point='gym_gazebo_ros.envs.tiago_robot:TiagoPickEnv',
    max_episode_steps=1000,
    reward_threshold=4.1,
)