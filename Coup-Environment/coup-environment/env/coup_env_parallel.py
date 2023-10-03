import functools
import random

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

# need to have stealing and blocking directed at a player
# assassinations need to remove influence
# deck needs not to be simply random
# exchange should pick the best of 4 cards
# upon losing influence, choose a card to discard
# upon winning a challenge, swap this card out

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


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "coup_v0"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # deprecated?
        # self._action_spaces = {agent: Discrete(2) for agent in self.possible_agents}
        # self._observation_spaces = {
        #     agent: Discrete(3) for agent in self.possible_agents
        # }
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
            turn = self.state[self.agents[0]]["TURN"]
            string = "{}: {}, coins {}, cards {}".format(
                turn,
                MOVES[self.state[turn]["MOVES"][-1]],
                self.state[turn]["COINS"],
                self.state[turn]["CARDS"],
            )
        else:
            string = "Game over"
        print(string)

    # come back to this after reset
    def observe(self, agent):
        return np.array(self.observations[agent])

    def reset(self, seed=None, options=None):
        # self.rewards = {agent: 0 for agent in self.agents}
        # self._cumulative_rewards = {agent: 0 for agent in self.agents}
        # self.terminations = {agent: False for agent in self.agents}
        # self.truncations = {agent: False for agent in self.agents}
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {
            agent: {
                "COINS": 0,
                "MOVES": [11],
                "CARDS": [genCard(), genCard()],
                "TURN": "player_0",
            }
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        self.state = observations
        return observations, infos

    def step(self, actions):
        def checkCard(card, agent, reward=5):
            for i in self.agents:
                if agent == i:
                    continue
                has_card = card in self.state[i]["CARDS"]
                if has_card:
                    rewards[i] = reward
                    rewards[agent] = -reward
                else:
                    rewards[i] = -reward
                    rewards[agent] = reward
                return has_card

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        rewards = {}
        for agent, action in actions.items():
            self.state[agent]["MOVES"].append(action)
            # BLOCK FOREIGN AID
            if action == 7:
                for i in self.agents:
                    if self.state[i]["MOVES"][-1] != 1:
                        continue
                    self.state[i]["COINS"] -= 2
                    rewards[i] = -2
                    rewards[agent] = 2
            # BLOCK STEALING
            elif action == 8:
                for i in self.agents:
                    if self.state[i]["MOVES"][-1] != 6:
                        continue
                    self.state[i]["COINS"] -= 2
                    self.state[agent]["COINS"] += 2
                    rewards[agent] = 2
                    rewards[i] = -2
            # BLOCK ASSASSINATION
            elif action == 9:
                for i in self.agents:
                    if self.state[i]["MOVES"][-1] != 4:
                        continue
                    rewards[agent] = 5
                    rewards[i] = -5

            # turn starts here
            if agent != self.state[agent]["TURN"]:
                continue

            # INCOME
            if action == 0:
                self.state[agent]["COINS"] += 1
                rewards[agent] = 1
            # FOREIGN AID
            elif action == 1:
                self.state[agent]["COINS"] += 2
                rewards[agent] = 2
            # COUP
            elif action == 2 and self.state[agent]["COINS"] >= 7:
                self.state[agent]["COINS"] -= 7
                rewards[agent] = 5
            # TAX
            elif action == 3:
                self.state[agent]["COINS"] += 3
                rewards[agent] = 3
            # ASSASSINATE
            elif action == 4 and self.state[agent]["COINS"] >= 3:
                self.state[agent]["COINS"] -= 3
                rewards[agent] = 5
            # EXCHANGE
            elif action == 5:
                self.state[agent]["CARDS"] = [genCard(), genCard()]
            # STEAL
            elif action == 6:
                self.state[agent]["COINS"] += 2
                for i in self.agents:
                    if agent == i:
                        continue
                    self.state[i]["COINS"] -= 2
                    rewards[i] = -2
                rewards[agent] = 2
            # CHALLENGE
            elif action == 10:
                for i in self.agents:
                    if agent == i:
                        continue
                    last_move = self.state[i]["MOVES"][-1]
                    if last_move in [3, 7]:
                        checkCard("DUKE", agent)
                    elif last_move == 4:
                        checkCard("ASSASSIN", agent)
                    elif last_move == 5:
                        checkCard("AMBASSADOR", agent)
                    elif last_move == 6:
                        checkCard("CAPTAIN", agent)
                    elif last_move == 8:
                        if checkCard("CAPTAIN", agent, reward=0):
                            checkCard("CAPTAIN", agent)
                        if checkCard("AMBASSADOR", agent, reward=0):
                            checkCard("AMBASSADOR", agent)
                    elif last_move == 9:
                        checkCard("CONTESSA", agent)

        terminations = {agent: False for agent in self.agents}

        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        turn = self.state[self.agents[0]]["TURN"]
        next_turn = ""
        if turn == "player_0":
            next_turn = "player_1"
        else:
            next_turn = "player_0"

        observations = {}
        for i in self.state:
            self.state[i]["TURN"] = next_turn
        for i in self.agents:
            observations[i] = self.state

        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos
