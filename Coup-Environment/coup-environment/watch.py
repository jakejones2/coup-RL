import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from custom_model import Model1
from env.coup_env import CoupFourPlayers

ModelCatalog.register_custom_model("Model1", Model1)

env = CoupFourPlayers(render_mode="all")
register_env(
    "CoupFourPlayers",
    lambda config: PettingZooEnv(CoupFourPlayers(render_mode="human")),
)

ray.init()

PPOagent = PPO.from_checkpoint("checkpoint/path/goes/here")

observations, infos = env.reset()

rewards_sum = {}

while env.agents:
    actions = {
        agent: PPOagent.compute_single_action(observations[agent])
        for agent in env.agents
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)
    for agent, reward in rewards.items():
        rewards_sum[agent] += reward
env.close()

print("Total Rewards: ", rewards_sum)
