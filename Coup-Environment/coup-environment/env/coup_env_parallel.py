import functools
import random

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium import spaces

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

# need to have stealing and blocking directed at a player
# need to ensure only one counteraction or block per move
# deck needs not to be simply random
# add mask to prevent illegal moves?
# enforce discard after challenge

MOVES = [
    "INCOME",  # 0
    "FOREIGN AID",  # 1
    "COUP",  # 2
    "TAX",  # 3
    "ASSASSINATE",  # 4
    "EXCHANGE",  # 5
    "STEAL",  # 6
    "BLOCK_FOREIGN_AID",  # 7
    "BLOCK_STEALING",  # 8
    "BLOCK_ASSASSINATION",  # 9
    "CHALLENGE",  # 10
    "DISCARD_DUKE",  # 11
    "DISCARD_CONTESSA",  # 12
    "DISCARD_CAPTAIN",  # 13
    "DISCARD_AMBASSADOR",  # 14
    "DISCARD_ASSASSIN",  # 15
    "None",  # 16
]
NUM_ITERS = 100
CARDS = [
    "DUKE",  # 0
    "ASSASSIN",  # 1
    "AMBASSADOR",  # 2
    "CAPTAIN",  # 3
    "CONTESSA",  # 4
    "None",  # 5
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
        self._agent_ids = {"player_0", "player_1"}
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        self.rewards_log = {}
        self.observation_spaces = {
            agent: spaces.Tuple(
                (
                    Discrete(2),  # turn
                    Discrete(100),  # player coins
                    Discrete(100),  # opponent coins
                    Discrete(6),  # card 1
                    Discrete(6),  # card 2
                    Discrete(6),  # card 3
                    Discrete(6),  # card 4
                    Discrete(17),  # opponent last move
                    Discrete(17),  # opponent second last move
                    Discrete(17),  # opponent third last move
                )
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {agent: Discrete(16) for agent in self.possible_agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:
            turn = self.state["player_0"]["TURN"]
            string = "{}: {}, coins {}, cards {}".format(
                turn,
                MOVES[self.state[turn]["MOVES"][-1]],
                self.state[turn]["COINS"],
                self.state[turn]["CARDS"],
            )
            # render any counteractions
            counter = self.agents[self.agents.index(self.state["player_0"]["TURN"]) - 1]
            if self.state[counter]["MOVES"][-1] in [7, 8, 9, 10]:
                string += "\ncounter from {}: {}, coins {}, cards {}".format(
                    counter,
                    MOVES[self.state[counter]["MOVES"][-1]],
                    self.state[counter]["COINS"],
                    self.state[counter]["CARDS"],
                )
            string += "\n    >  " + str(self.rewards_log)
        else:
            string = "Game over"
        print(string)

    # come back to this after reset
    # def observe(self, agent):
    #     return np.array(self.observations[agent])

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents
        self.num_moves = 0
        self.state = {
            agent: {
                "COINS": 0,
                "MOVES": [16, 16, 16],
                "CARDS": [genCard(), genCard(), "None", "None"],
                "TURN": "player_0",
            }
            for agent in self.agents
        }
        observations = {
            agent: (
                1 if self.state[agent]["TURN"] == agent else 0,
                0,
                0,
                CARDS.index(self.state[agent]["CARDS"][0]),
                CARDS.index(self.state[agent]["CARDS"][1]),
                5,
                5,
                16,
                16,
                16,
            )
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        def checkCard(card, agent, opponent, reward=5):
            has_card = card in self.state[opponent]["CARDS"]
            if has_card:
                rewards[opponent] = reward
                rewards[agent] = -reward
            else:
                rewards[opponent] = -reward
                rewards[agent] = reward
            return has_card

        # def failedChallenge(agent, opponent):
        #     moves = self.state[opponent]["MOVES"].copy().reverse()
        #     for move in moves:
        #         if move < 7:
        #             return move

        def removeCard(card, agent):
            """
            If agent doesn't hold card, punish 10.
            If agent holds card, remove card from state.
            """
            if not card in self.state[agent]["CARDS"]:
                rewards[agent] -= 10
                return
            self.state[agent]["CARDS"][self.state[agent]["CARDS"].index(card)] = "None"

        def lastTurn(player):
            """
            Return a player's last move, discounting counters, discards and challenges.
            """
            moves = self.state[player]["MOVES"].copy().reverse()
            if not moves:
                return 16
            for move in moves:
                if move < 7:
                    return move

        rewards = {}
        for i in self.agents:
            rewards[i] = 0
        observations = {}
        infos = {}

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        for agent, action in actions.items():
            # handle turn-based actions
            if agent != self.state[agent]["TURN"] and action < 7:
                rewards[agent] -= 10
                continue

            self.state[agent]["MOVES"].append(action)
            opponent = "player_0" if agent == "player_1" else "player_1"

            # punish lack of discard if applicable
            if (
                len(self.state[agent]["CARDS"]) > 2 or lastTurn(opponent) in [2, 4]
            ) and (action < 11 and action != 9):
                rewards[agent] -= 10

            match action:
                case 0:  # INCOME
                    self.state[agent]["COINS"] += 1
                    rewards[agent] += 1
                case 1:  # FOREIGN AID
                    self.state[agent]["COINS"] += 2
                    rewards[agent] += 2
                case 2:  # COUP
                    if self.state[agent]["COINS"] >= 7:
                        self.state[agent]["COINS"] -= 7
                        rewards[agent] += 5
                        rewards[opponent] -= 5
                case 3:  # TAX
                    self.state[agent]["COINS"] += 3
                    rewards[agent] += 3
                case 4:  # ASSASSINATE
                    if self.state[agent]["COINS"] >= 3:
                        self.state[agent]["COINS"] -= 3
                        rewards[agent] -= 5
                        rewards[opponent] -= 5
                case 5:  # EXCHANGE
                    try:
                        cards = self.state[agent]["CARDS"]
                        self.state[agent]["CARDS"][cards.index("None")] = genCard()
                        self.state[agent]["CARDS"][cards.index("None")] = genCard()
                    except ValueError:
                        rewards[agent] -= 3
                case 6:  # STEAL
                    self.state[agent]["COINS"] += 2
                    self.state[opponent]["COINS"] -= 2
                    rewards[agent] += 2
                    rewards[opponent] -= 2
                case 7:  # BLOCK FOREIGN AID
                    if lastTurn(opponent) != 1:
                        rewards[agent] -= 10
                    else:
                        self.state[opponent]["COINS"] -= 2
                        rewards[opponent] -= 2
                        rewards[agent] += 2
                case 8:  # BLOCK STEALING
                    if lastTurn(opponent) != 6:
                        rewards[agent] -= 10
                    else:
                        self.state[opponent]["COINS"] -= 2
                        self.state[agent]["COINS"] += 2
                        rewards[opponent] -= -2
                        rewards[agent] += 2
                case 9:  # BLOCK ASSASSINATION
                    if lastTurn(opponent) != 4:
                        rewards[agent] -= 10
                    else:
                        rewards[opponent] -= 5
                        rewards[agent] += 5
                case 10:  # CHALLENGE
                    last_move = self.state[opponent]["MOVES"][-1]
                    if last_move in [3, 7]:
                        checkCard("DUKE", agent, opponent)
                    elif last_move == 4:
                        checkCard("ASSASSIN", agent, opponent)
                    elif last_move == 5:
                        checkCard("AMBASSADOR", agent, opponent)
                    elif last_move == 6:
                        checkCard("CAPTAIN", agent, opponent)
                    elif last_move == 8:
                        if checkCard("CAPTAIN", agent, opponent, reward=0):
                            checkCard("CAPTAIN", agent, opponent)
                        if checkCard("AMBASSADOR", agent, opponent, reward=0):
                            checkCard("AMBASSADOR", agent, opponent)
                    elif last_move == 9:
                        checkCard("CONTESSA", agent, opponent)
                case 11:  # DISCARD DUKE
                    removeCard("DUKE", agent)
                case 12:  # DISCARD CONTESSA
                    removeCard("CONTESSA", agent)
                case 13:  # DISCARD CAPTAIN
                    removeCard("CAPTAIN", agent)
                case 14:  # DISCARD AMBASSADOR
                    removeCard("AMBASSADOR", agent)
                case 15:  # DISCARD ASSASSIN
                    removeCard("ASSASSIN", agent)

            observations[agent] = (
                1 if self.state[agent]["TURN"] == agent else 0,
                self.state[agent]["COINS"],
                self.state[opponent]["COINS"],
                CARDS.index(self.state[agent]["CARDS"][0]),
                CARDS.index(self.state[agent]["CARDS"][1]),
                CARDS.index(self.state[agent]["CARDS"][2]),
                CARDS.index(self.state[agent]["CARDS"][3]),
                self.state[opponent]["MOVES"][-1],
                self.state[opponent]["MOVES"][-2],
                self.state[opponent]["MOVES"][-3],
            )

            infos[agent] = {}

        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS

        terminations = {
            agent: len(self.state[agent]["CARDS"]) == 0 for agent in self.agents
        }
        truncations = {agent: env_truncation for agent in self.agents}

        env_termination = len(
            [item for item in terminations.items() if item[1] == True]
        )

        self.rewards_log = rewards

        if self.render_mode == "human":
            self.render()

        turn = self.state[self.agents[0]]["TURN"]
        next_turn = ""
        if turn == "player_0":
            next_turn = "player_1"
        else:
            next_turn = "player_0"

        for i in self.state:
            self.state[i]["TURN"] = next_turn

        if env_truncation or env_termination:
            self.agents = []

        return observations, rewards, terminations, truncations, infos
