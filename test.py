"""
Script for testing trained policies against a random policy.
Plays multiple games and calculates averages.
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

raw_env = CoupFourPlayers(render_mode="none")
env = ParallelPettingZooEnv(raw_env)
env._agent_ids = set(raw_env.agents)
register_env("coup_env", lambda config: env)

ray.init()

PPOagent = PPO.from_checkpoint(
    "/Users/jakejones/Documents/repos/git/petting-zoo/ray_results/coup_env/PPO/PPO_coup_env_5b9db_00000_0_2023-10-23_09-30-01/checkpoint_000209"
    # policy checkpoint goes here
)


def test_policy(player, iter, results, label=""):
    if not label:
        label = player
    for n in range(iter):
        observations, infos = env.reset()
        reward_sum = {agent: 0 for agent in raw_env.agents}
        while True:
            actions = {
                "player_0": random_policy(observations["player_0"]),
                "player_1": random_policy(observations["player_1"]),
                "player_2": random_policy(observations["player_2"]),
                "player_3": random_policy(observations["player_3"]),
            }
            actions[player] = PPOagent.compute_single_action(
                observations[player],  # policy_id="main"
            )
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent, reward in rewards.items():
                reward_sum[agent] += reward
            if n % 10 == 0:
                loader = "testing" + ("." * int(n / 10))
                print(loader, end="\r")
            if terminations["__all__"] or truncations["__all__"]:
                reward_list = sorted(reward_sum.items(), key=lambda a: a[1])
                winner = [agent for agent, reward in reward_list][-1]
                if winner == player:
                    results["policy"] += 1
                else:
                    results["random"] += 1
                break
    print(f"batch {label} complete!")


test_results = {"policy": 0, "random": 0}
print("\n----Policy Testing----\n")
test_policy("player_0", 100, test_results, 1)
test_policy("player_1", 100, test_results, 2)
test_policy("player_2", 100, test_results, 3)
test_policy("player_3", 100, test_results, 4)
policy_wins = test_results["policy"]
random_wins = test_results["random"]
policy_win_rate = round(policy_wins * 100 / (policy_wins + random_wins), 3)
print("\nResults:\n")
print(f"Trained policy wins: {policy_wins}, Random policy wins: {random_wins}")
print(f"Trained policy wins {policy_win_rate}% of the time.")
print(f"Trained policy is {policy_win_rate - 25}% better than average.")
print("\n----End Policy Testing----\n")


env.close()

# RAY_DEDUP_LOGS=0 py watch.py
