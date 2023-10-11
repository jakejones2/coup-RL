from env import coup_env_multiplayer
import random

env = coup_env_multiplayer.parallel_env(render_mode="human")
observations, infos = env.reset()


while env.agents:

    def select_action(agent):
        mask = list(observations[agent]["action_mask"])
        # print("mask", mask)
        actions = []
        for n in range(26):
            if mask[n]:
                actions.append(n)
        if len(actions) == 0:
            actions = [25]
        # print("actions", actions)
        return random.choice(actions)

    actions = {agent: select_action(agent) for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    # print(observations)
    # print(rewards)
env.close()
