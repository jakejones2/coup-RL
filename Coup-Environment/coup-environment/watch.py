import argparse
import os

import ray

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from ml import Model1

from env.coup_env_parallel import parallel_env

# parser = argparse.ArgumentParser(
#     description="Render pretrained policy loaded from checkpoint"
# )
# parser.add_argument(
#     "--checkpoint-path",
#     help="Path to the checkpoint",
# )

# args = parser.parse_args()

# checkpoint_path = os.path.expanduser(args.checkpoint_path)

ModelCatalog.register_custom_model("Model1", Model1)

env = parallel_env(render_mode="all")
register_env(
    "coup_env_parallel", lambda config: PettingZooEnv(parallel_env(render_mode="all"))
)

ray.init()

PPOagent = PPO.from_checkpoint(
    "/Users/jakejones/Documents/repos/git/petting-zoo/ray_results/coup_env_parallel/PPO/PPO_coup_env_parallel_9300a_00000_0_2023-10-09_14-13-01/checkpoint_000096"
)

env.reset()

reward_sum1 = 0
reward_sum2 = 0

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    # print(observations)
    # print(rewards)
    reward_sum1 += rewards["player_0"]
    reward_sum2 += rewards["player_1"]
env.close()
