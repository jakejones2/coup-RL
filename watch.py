"""
Script for watching trained policies play against a random policy.
Plays one game.
"""

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.tune.registry import register_env

from model import Model1
from env.coup_env import CoupFourPlayers
from random_policy import random_policy

tf1, tf, tfv = try_import_tf()
tf1.enable_eager_execution()

ModelCatalog.register_custom_model("Model1", Model1)

raw_env = CoupFourPlayers(render_mode="moves")
env = ParallelPettingZooEnv(raw_env)
env._agent_ids = set(raw_env.agents)
register_env("coup_env", lambda config: env)

ray.init()

PPOagent = PPO.from_checkpoint(
    "/Users/jakejones/Documents/repos/git/petting-zoo/ray_results/coup_env/PPO/PPO_coup_env_f9312_00000_0_2023-10-23_19-42-52/checkpoint_000192"
    # policy checkpoint goes here
)

observations, infos = env.reset()

while True:
    actions = {
        "player_0": PPOagent.compute_single_action(
            observations["player_0"], policy_id="main"
        ),
        "player_1": random_policy(observations["player_1"]),
        "player_2": random_policy(observations["player_2"]),
        "player_3": random_policy(observations["player_3"]),
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)
    if terminations["__all__"] or truncations["__all__"]:
        break
env.close()

# RAY_DEDUP_LOGS=0 py watch.py
