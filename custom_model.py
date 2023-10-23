import os

from gymnasium.spaces import Dict

import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.rllib.utils import check_env
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork

import tensorflow as tf

import supersuit as ss

from env.coup_env import CoupFourPlayers


class Model1(TFModelV2):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )
        super(Model1, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        # action_space is Discrete(26)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)
        self.no_masking = False

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        print("action mask here --> ", action_mask)
        print("observation here ->", input_dict["obs"]["observations"])
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})
        print("unmasked logits here -> ", logits)
        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask
        print("masked_logits here -> ", masked_logits)
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


if __name__ == "__main__":

    def env_creator(args):
        env = CoupFourPlayers(render_mode="human")
        # supersuit cannot handle the dict type observation with action_mask key - expects np.array
        # env = ss.dtype_v0(env, "float32")
        # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
        wrapped_env = ParallelPettingZooEnv(env)
        # this shouldn't be neccessary - some sort of bug here?
        # https://github.com/ray-project/ray/blob/master/rllib/env/wrappers/pettingzoo_env.py
        wrapped_env._agent_ids = set(env.agents)
        return wrapped_env

    ray.init()

    env_name = "coup_env_parallel_logs"
    register_env(env_name, lambda config: env_creator(config))
    ModelCatalog.register_custom_model("Model1", Model1)

    config = (
        PPOConfig()
        .rl_module(_enable_rl_module_api=False)
        .environment(
            env=env_name,
            disable_env_checking=False,
            action_mask_key="action_mask",
        )  # clip_actions=True
        .rollouts(num_rollout_workers=1, rollout_fragment_length=128)  # workers = 4
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
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
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
