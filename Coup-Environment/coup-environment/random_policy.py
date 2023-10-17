from env.coup_env import CoupFourPlayers
import random

env = CoupFourPlayers(render_mode="human")
observations, infos = env.reset()

while env.agents:

    def select_action(agent):
        mask = list(observations[agent]["action_mask"])
        actions = []
        for n in range(26):
            if mask[n]:
                actions.append(n)
        if len(actions) == 0:
            actions = [25]
        return random.choice(actions)

    actions = {agent: select_action(agent) for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
