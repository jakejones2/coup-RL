import functools
import random

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

# to make this 3 or more players, need to use the parallel API which can handle counteractions

MOVES = [
    "INCOME",
    "FOREIGN AID",
    "COUP",
    "TAX",
    "ASSASSINATE",
    "EXCHANGE",
    "STEAL",
    "BLOCK FOREIGN AID",
    "BLOCK STEALING",
    "BLOCK ASSASSINATION",
    "CHALLENGE",
    "None",
]
NUM_ITERS = 100
CARDS = [
    "DUKE",
    "ASSASSIN",
    "AMBASSADOR",
    "CAPTAIN",
    "CONTESSA",
]


def genCard():
    return CARDS[random.randint(0, 4)]


def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "coup_v0"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # learn this
        self._action_spaces = {agent: Discrete(2) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Discrete(3) for agent in self.possible_agents
        }
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Discrete(3)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(2)

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:
            string = "Current state: Agent1: move {} coins {} , Agent2: move {} coins {}".format(
                MOVES[self.state[self.agents[0]]["MOVE"]],
                self.state[self.agents[0]]["COINS"],
                MOVES[self.state[self.agents[1]]["MOVE"]],
                self.state[self.agents[1]]["COINS"],
            )
        else:
            string = "Game over"
        print(string)

    # come back to this after reset
    def observe(self, agent):
        return np.array(self.observations[agent])

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {
            agent: {"COINS": 0, "MOVES": [11], "CARDS": [genCard(), genCard()]}
            for agent in self.agents
        }
        self.observations = {agent: 11 for agent in self.agents}
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        # self._cumulative_rewards[agent] = 0
        self.state[agent]["MOVE"] = action

        self.num_moves += 1

        # INCOME
        if action == 0:
            self.state[agent]["COINS"] += 1
            self.rewards[agent] = 1
        # ASSASSINATE
        elif action == 1 and self.state[agent]["COINS"] >= 3:
            self.state[agent]["COINS"] -= 3
            self.rewards[agent] = 3

        self.truncations = {agent: self.num_moves >= NUM_ITERS for agent in self.agents}

        for i in self.agents:
            self.observations[i] = self.state

        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()
