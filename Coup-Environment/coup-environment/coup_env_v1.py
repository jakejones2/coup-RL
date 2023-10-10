from env import coup_env_multiplayer

env = coup_env_multiplayer.parallel_env(render_mode="all")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    print(observations)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    # print(observations)
    # print(rewards)
env.close()
