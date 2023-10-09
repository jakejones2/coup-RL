import os

import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.rllib.utils import check_env

from keras.layers import Dense
from keras.models import Sequential

from env import coup_env_parallel


class Model1(TFModelV2):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TFModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        self.model = Sequential()
        self.model.add(Dense(10, input_shape=(1, 10), activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(16, activation="softmax"))

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.model(input_dict["obs"])
        return model_out, state

    # def value_function(self):
    #     """Something goes here"""


def env_creator(args):
    env = coup_env_parallel.parallel_env(render_mode="human")
    # some super suit functions went here, normalising observations and stacking frames
    # check_env(env)
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "coup_env_parallel"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("Model1", Model1)

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True, disable_env_checking=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)  # workers = 4
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
        )
        .debugging(log_level="ERROR")  # bad methods?
        .framework(framework="tf")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 500000},  # 5000000
        checkpoint_freq=10,
        local_dir="/Users/jakejones/Documents/repos/git/petting-zoo/ray_results/"
        + env_name,
        config=config.to_dict(),
    )
