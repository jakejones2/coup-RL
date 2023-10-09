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
# need to enforce waiting for exchange to complete...

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

CARDS = [
    "DUKE",  # 0
    "ASSASSIN",  # 1
    "AMBASSADOR",  # 2
    "CAPTAIN",  # 3
    "CONTESSA",  # 4
    "None",  # 5
]

NUM_ITERS = 100


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
        self.rewards = {}
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
        agent = self.state["player_0"]["TURN"]
        opponent = "player_1" if agent == "player_0" else "player_0"
        agent_moves = self.state[agent]["MOVES"]
        opponent_moves = self.state[opponent]["MOVES"]
        last_move = agent_moves[-1]
        last_counter = opponent_moves[-1]

        # check for failed challenge resulting in automatic coup
        if agent_moves[-1] == 2 and (agent_moves[-2] == 10 or opponent_moves[-1] == 10):
            last_move = agent_moves[-2]

        if opponent_moves[-1] == 2 and (
            agent_moves[-2] == 10 or opponent_moves[-1] == 10
        ):
            last_counter = opponent_moves[-2]

        # render the player who's turn it is (agent)
        string = "{}: {}, coins {}, cards {}".format(
            agent,
            MOVES[last_move],
            self.state[agent]["COINS"],
            self.state[agent]["CARDS"],
        )

        # render any counteractions (opponent)
        if self.render_mode == "all" or last_counter in [
            7,
            8,
            9,
            10,
        ]:
            string += "\ncounter from {}: {}, coins {}, cards {}".format(
                opponent,
                MOVES[last_counter],
                self.state[opponent]["COINS"],
                self.state[opponent]["CARDS"],
            )
            if self.render_mode == "all":
                string += "\n    >  " + str(self.rewards)
        print(string)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents
        self.num_moves = 0
        self.rewards = {}
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

    def has_card(self, card, player):
        """
        Returns true if player holds card, else false
        """
        return card in self.state[player]["CARDS"]

    def resolve_challenge(
        self,
        card,
        agent,
        opponent,
    ):
        """
        Reward players if they pass/fail challenge
        """
        if self.has_card(card, opponent):
            if card == "ASSASSIN":
                self.state[agent]["CARDS"] = ["None", "None", "None", "None"]
            else:
                self.rewards[opponent] = 5
                self.rewards[agent] = -5
                self.state[opponent]["MOVES"].append(2)
        else:
            self.rewards[opponent] = -5
            self.rewards[agent] = 5
            self.state[agent]["MOVES"].append(2)

    def remove_card(self, card, agent):
        """
        If agent doesn't hold card, punish 10.
        If agent holds card, remove card from state.
        """
        if not card in self.state[agent]["CARDS"]:
            self.rewards[agent] -= 10
            return
        self.state[agent]["CARDS"][self.state[agent]["CARDS"].index(card)] = "None"

    def last_turn(self, player):
        """
        Return a player's last turn-based move, discounting counters, discards, challenges and None.
        If no moves, return None.
        """
        moves = self.state[player]["MOVES"].copy().reverse()
        if not moves:
            return 16
        for move in moves:
            if move < 7:
                return move
        return 16

    def last_move(self, player):
        """
        Return a player's last action, discounting None.
        If no moves, return None.
        """
        moves = self.state[player]["MOVES"].copy().reverse()
        if not moves:
            return 16
        for move in moves:
            if move != 16:
                return move
        return 16

    def number_of_cards(self, player):
        """
        Returns the number of cards a player has discounting None.
        """
        cards = self.state[player]["CARDS"]
        count = 0
        for card in cards:
            if card != "None":
                count += 1
        return count

    def step(self, actions):
        self.rewards = {}
        observations = {}
        infos = {}

        for i in self.agents:
            self.rewards[i] = 0

        for agent, action in actions.items():
            opponent = "player_0" if agent == "player_1" else "player_1"

            # punish incorrect turns
            if agent != self.state[agent]["TURN"] and action < 7:
                self.rewards[agent] -= 10
                continue

            # punish lack of discard if applicable
            if (
                (
                    self.number_of_cards(agent) > 2 or self.last_turn(agent) == 5
                )  # if exchanged
                and action < 11
            ) or (
                self.last_turn(opponent) in [2, 4] and action < 9
            ):  # or assassinated/couped
                self.rewards[agent] -= 10
                continue

            # punish lack of wait for discard by opponent
            if (
                (
                    self.last_turn(agent) in [2, 4]
                    and self.state[opponent]["MOVES"][-1] < 9
                )  # if assassinating/launching coup
                or self.number_of_cards(opponent) > 2
                or self.last_turn(opponent) == 5
                # or opponent exchanging
            ) and action != 16:
                self.rewards[agent] -= 10
                continue

            self.state[agent]["MOVES"].append(action)

            match action:
                case 0:  # INCOME
                    self.state[agent]["COINS"] += 1
                    self.rewards[agent] += 1
                case 1:  # FOREIGN AID
                    self.state[agent]["COINS"] += 2
                    self.rewards[agent] += 2
                case 2:  # COUP
                    if self.state[agent]["COINS"] >= 7:
                        self.state[agent]["COINS"] -= 7
                        self.rewards[agent] += 5
                        self.rewards[opponent] -= 5
                    else:
                        self.rewards[agent] -= 10
                case 3:  # TAX
                    self.state[agent]["COINS"] += 3
                    self.rewards[agent] += 3
                case 4:  # ASSASSINATE
                    if self.state[agent]["COINS"] >= 3:
                        self.state[agent]["COINS"] -= 3
                        self.rewards[agent] -= 5
                        self.rewards[opponent] -= 5
                case 5:  # EXCHANGE
                    try:
                        cards = self.state[agent]["CARDS"]
                        self.state[agent]["CARDS"][cards.index("None")] = genCard()
                        if self.number_of_cards(agent) > 2:
                            self.state[agent]["CARDS"][cards.index("None")] = genCard()
                    except ValueError:
                        self.rewards[agent] -= 3
                case 6:  # STEAL
                    if self.state[opponent]["COINS"] >= 2:
                        self.state[agent]["COINS"] += 2
                        self.state[opponent]["COINS"] -= 2
                        self.rewards[agent] += 2
                        self.rewards[opponent] -= 2
                    elif self.state[opponent]["COINS"] == 1:
                        self.state[agent]["COINS"] += 1
                        self.state[opponent]["COINS"] -= 1
                        self.rewards[agent] += 1
                        self.rewards[opponent] -= 1
                case 7:  # BLOCK FOREIGN AID
                    if self.last_turn(opponent) != 1:
                        self.rewards[agent] -= 10
                    else:
                        self.state[opponent]["COINS"] -= 2
                        self.rewards[opponent] -= 2
                        self.rewards[agent] += 2
                case 8:  # BLOCK STEALING
                    if self.last_turn(opponent) != 6:
                        self.rewards[agent] -= 10
                    else:
                        self.state[opponent]["COINS"] -= 2
                        self.state[agent]["COINS"] += 2
                        self.rewards[opponent] -= -2
                        self.rewards[agent] += 2
                case 9:  # BLOCK ASSASSINATION
                    if self.last_turn(opponent) != 4:
                        self.rewards[agent] -= 10
                    else:
                        self.rewards[opponent] -= 5
                        self.rewards[agent] += 5
                case 10:  # CHALLENGE
                    last_move = self.state[opponent]["MOVES"][-1]
                    if last_move in [3, 7]:
                        self.resolve_challenge("DUKE", agent, opponent)
                    elif last_move == 4:
                        self.resolve_challenge("ASSASSIN", agent, opponent)
                    elif last_move == 5:
                        self.resolve_challenge("AMBASSADOR", agent, opponent)
                    elif last_move == 6:
                        self.resolve_challenge("CAPTAIN", agent, opponent)
                    elif last_move == 8:
                        if self.has_card("CAPTAIN", opponent):
                            self.resolve_challenge("CAPTAIN", agent, opponent)
                        else:
                            self.resolve_challenge("AMBASSADOR", agent, opponent)
                    elif last_move == 9:
                        self.resolve_challenge("CONTESSA", agent, opponent)
                    else:
                        self.rewards[agent] = -10
                case 11:  # DISCARD DUKE
                    self.remove_card("DUKE", agent)
                case 12:  # DISCARD CONTESSA
                    self.remove_card("CONTESSA", agent)
                case 13:  # DISCARD CAPTAIN
                    self.remove_card("CAPTAIN", agent)
                case 14:  # DISCARD AMBASSADOR
                    self.remove_card("AMBASSADOR", agent)
                case 15:  # DISCARD ASSASSIN
                    self.remove_card("ASSASSIN", agent)
                case 16:  # None
                    pass

            # ensure coins remain positive
            if self.state[agent]["COINS"] < 0:
                self.state[agent]["COINS"] = 0

            if self.state[opponent]["COINS"] < 0:
                self.state[opponent]["COINS"] = 0

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
            agent: self.number_of_cards(agent) == 0 for agent in self.agents
        }
        truncations = {agent: env_truncation for agent in self.agents}

        env_termination = len(
            [item for item in terminations.items() if item[1] == True]
        )

        if self.render_mode in ["human", "all"]:
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
            print("Game Over")
            self.agents = []

        return observations, self.rewards, terminations, truncations, infos
