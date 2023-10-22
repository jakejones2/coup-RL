import os

import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune

from env.coup_env import CoupFourPlayers


def env_creator(args):
    env = CoupFourPlayers(render_mode="none")
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "CoupFourPlayers"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True, disable_env_checking=True)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=128)  # workers = 4
        .rl_module(_enable_rl_module_api=True)
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
            _enable_learner_api=True,
        )
        .debugging(log_level="ERROR")
        .framework(framework="tf")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 2},  # 5000000
        checkpoint_freq=10,
        local_dir="/Users/jakejones/Documents/repos/git/petting-zoo/ray_results/"
        + env_name,
        config=config.to_dict(),
    )