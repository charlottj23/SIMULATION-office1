from envs.custom_mujoco_env import CustomRobotEnv
from envs.world_env import CustomWorldEnv
from gymnasium.envs.registration import register

register(
    id="CustomRobot-v0",
    entry_point="envs.custom_mujoco_env:CustomRobotEnv",
    max_episode_steps=200,
)

register(
    id="CustomWorld-v0",
    entry_point="envs.world_env:CustomWorldEnv",
    max_episode_steps=250,
)

__all__ = ["CustomRobotEnv", "CustomWorldEnv"]


