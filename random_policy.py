"""
Random policy chooses random actions after filtering out
illegal moves using the action mask.

Run this script to see a single game.
"""

from env.coup_env import CoupFourPlayers
import random


def random_policy(observation):
    mask = list(observation["action_mask"])
    actions = []
    for n in range(26):
        if mask[n]:
            actions.append(n)
    if len(actions) == 0:
        actions = [25]
    return random.choice(actions)


if __name__ == "__main__":
    env = CoupFourPlayers(render_mode="moves")
    observations, infos = env.reset()

    while True:
        actions = {agent: random_policy(observations[agent]) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        if terminations["__all__"] or truncations["__all__"]:
            break

    env.close()
