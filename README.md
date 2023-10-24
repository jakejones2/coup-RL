# Coup-RL

Training a keras model to play the card game **[Coup](https://www.ultraboardgames.com/coup/game-rules.php)** using Reinforcement Learning. This is a work in progress!

### Setup

To see the environment in action, run `random_policy.py`. This plays the game with random decisions by each agent.
To train policies, experiment with the ppo files or create your own algorithms. [See ray docs](https://docs.ray.io/en/latest/rllib/index.html) for info.
To view a policy versus random actions, modify and run `watch.py`.
To test a policy's performance versus random actions, modify and run `test.py`.

### Environment - PettingZoo/Gymnasium

Environment is mostly ready - missing a few subtleties of play such as the correct reinburcement when blocking a steal of 1 coin. Also need to redesign to take a variable number of players from 2-6 (this should be quite simple). At some point I might simplify rewards as proportional to coins and inversely proportional to remaining players, with a bonus for winning. Just to see the impact on training! Would also make the code a lot more concise.

### Training - RLlib/TensorFlow

Bigger challenge has been the RLlib algorithms - see policy stats below for progress so far. Need to optimise the PPO algorithm and potentially restructure rewards for better performance (need to research benefits of normalisation for rewards etc.). Also need to work out how to access GPU for faster training. [Ray docs](https://github.com/ray-project/ray/blob/master/rllib/env/wrappers/pettingzoo_env.py) suggest that standard algorithms assume agent cooperation to maximise reward -
this is significant and needs to be addressed!

### Policy Stats:

#### PPO_coup_env_5b9db_00000_0_2023-10-23_09-30-01

- Trained for ~2 hours with ppo_default.py
- Won 2204 out of 1796, or **55.1%** overall (30% better than random)

#### PPO_coup_env_f9312_00000_0_2023-10-23_19-42-52

- Trained for ~12 hours with ppo_competitive.py
- Won 1294 out of 2706, or **32.35%** overall (7% better than random)
