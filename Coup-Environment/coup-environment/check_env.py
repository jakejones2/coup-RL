from ray.rllib.utils import check_env
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from env import coup_env_parallel

env = coup_env_parallel.parallel_env()


def env_creator(args):
    env = coup_env_parallel.parallel_env(render_mode="human")
    # some super suit functions went here, normalising observations and stacking frames
    return env


register_env(
    "coup_env_parallel", lambda config: ParallelPettingZooEnv(env_creator(config))
)

check_env(env)
