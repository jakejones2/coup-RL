import random

import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.annotations import override

import tree

from typing import (
    List,
    Union,
)
from ray.rllib.utils.typing import TensorStructType

from model import Model1
from env.coup_env import CoupFourPlayers


class MaskedRandomPolicy(RandomPolicy):
    @override(RandomPolicy)
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        *args,
        **kwargs,
    ):
        actions = []
        for obs in obs_batch:
            action_mask = obs[:26]
            pos_actions = []
            for n in range(26):
                if action_mask[n]:
                    pos_actions.append(n)
            if len(pos_actions) == 0:
                pos_actions = [25]
            actions.append(random.choice(pos_actions))
        return (
            actions,
            [],
            {},
        )


if __name__ == "__main__":

    def env_creator(config):
        env = CoupFourPlayers(render_mode="games")
        wrapped_env = ParallelPettingZooEnv(env)
        wrapped_env._agent_ids = set(env.agents)
        return wrapped_env

    ray.init()

    env_name = "coup_env"
    register_env(env_name, lambda config: env_creator(config))
    ModelCatalog.register_custom_model("Model1", Model1)

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "main" if f"player_{episode.episode_id % 4}" == agent_id else "random"

    config = (
        PPOConfig()
        .rl_module(_enable_rl_module_api=False)
        .environment(
            env=env_name,
            disable_env_checking=False,
            action_mask_key="action_mask",
        )  # clip_actions=True, render_env=True
        .rollouts(num_rollout_workers=1, rollout_fragment_length=128)  # workers = 4
        .multi_agent(
            policies={
                "main": PolicySpec(),
                "random": PolicySpec(policy_class=MaskedRandomPolicy),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main"],
        )
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
            model={"custom_model": "Model1", "vf_share_layers": True},
            _enable_learner_api=False,
        )
        .debugging(log_level="ERROR")
        .framework(framework="tf2")
        .resources(num_gpus=0)
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 1},  # 5000000
        checkpoint_freq=10,
        local_dir="/Users/jakejones/Documents/repos/git/petting-zoo/ray_results/"
        + env_name,
        config=config.to_dict(),
    )

# RAY_DEDUP_LOGS=0 py custom_model.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/self_play_with_open_spiel.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/policy/random_policy.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_independent_learning.py
# https://github.com/PacktPublishing/Mastering-Reinforcement-Learning-with-Python/blob/master/Chapter10/mcar_train.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py
# https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#multi-agent
