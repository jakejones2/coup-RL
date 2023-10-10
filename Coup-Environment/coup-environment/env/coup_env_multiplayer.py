import functools
import random

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium import spaces

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

# make deck class with methods take, return, reset and shuffle
# go through and try optimize performance with numpy and manipulation of integers rather than strings

# need masks for various money things
# mask for assassinate and mask for coup
# mask for steal if others have money...

MOVES = [
    "COUP0",  # 0
    "COUP1",  # 1
    "COUP2",  # 2
    "COUP3",  # 3
    "ASSASSINATE0",  # 4
    "ASSASSINATE1",  # 5
    "ASSASSINATE2",  # 6
    "ASSASSINATE3",  # 7
    "STEAL0",  # 8
    "STEAL1",  # 9
    "STEAL2",  # 10
    "STEAL3",  # 11
    "INCOME",  # 12
    "FOREIGN AID",  # 13
    "TAX",  # 14
    "EXCHANGE",  # 15
    "BLOCK_FOREIGN_AID",  # 16
    "BLOCK_STEALING",  # 17
    "BLOCK_ASSASSINATION",  # 18
    "CHALLENGE",  # 19
    "DISCARD_DUKE",  # 20
    "DISCARD_CONTESSA",  # 21
    "DISCARD_CAPTAIN",  # 22
    "DISCARD_AMBASSADOR",  # 23
    "DISCARD_ASSASSIN",  # 24
    "None",  # 25
]

CARDS = [
    "DUKE",  # 0
    "CONTESSA",  # 1
    "CAPTAIN",  # 2
    "AMBASSADOR",  # 3
    "ASSASSIN",  # 4
    "None",  # 5
]

DECK = ["ASSASSIN", "AMBASSADOR", "DUKE", "CONTESSA", "CAPTAIN"] * 3
random.shuffle(DECK)

NUM_ITERS = 100

TURNS = [
    "player_0",
    "counter-player_0",
    "challenge-player_0",
    "discard-player_0",
    "player_1",
    "counter-player_1",
    "challenge-player_1",
    "discard-player_1",
    "player_2",
    "counter-player_2",
    "challenge-player_2",
    "discard-player_2",
    "player_3",
    "counter-player_3",
    "challenge-player_3",
    "discard-player_3",
]

leader_mask = np.pad(np.array([1, 1, 1]), (12, 11))
none_mask = np.zeros(26, "int8")
challenge_mask = np.append(np.pad(np.array([1]), (19, 5)), [1])
counter_fe_mask = np.append(np.pad(np.array([1, 0, 0, 1]), (16, 5)), [1])
counter_stealing_mask = np.append(np.pad(np.array([1, 0, 1]), (17, 5)), [1])
counter_assassin_mask = np.append(np.pad(np.array([1, 1]), (18, 5)), [1])
discard_mask = np.pad(np.array([1, 1, 1, 1, 1, 1]), (20, 0))


def reset_deck():
    DECK = ["ASSASSIN", "AMBASSADOR", "DUKE", "CONTESSA", "CAPTAIN"] * 3
    random.shuffle(DECK)


def take_card():
    return DECK.pop()


def gen_turn_list():
    round = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    rounds = []
    for n in range(100):
        rounds.extend(round)
    return rounds


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
        self.possible_agents = ["player_" + str(r) for r in range(4)]
        self._agent_ids = {"player_0", "player_1", "player_2", "player_3"}
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        self.rewards = {}
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Tuple(
                        (
                            Discrete(100),  # turn
                            Discrete(100),  # player0 coins
                            Discrete(100),  # player1 coins
                            Discrete(100),  # player2 coins
                            Discrete(100),  # player3 coins
                            Discrete(6),  # card 1
                            Discrete(6),  # card 2
                            Discrete(6),  # card 3
                            Discrete(6),  # card 4
                            Discrete(26),  # player0 last move
                            Discrete(26),  # player0 second last move
                            Discrete(26),  # player0 third last move
                            Discrete(26),  # player1 last move
                            Discrete(26),  # player1 second last move
                            Discrete(26),  # player1 third last move
                            Discrete(26),  # player2 last move
                            Discrete(26),  # player2 second last move
                            Discrete(26),  # player2 third last move
                            Discrete(26),  # player3 last move
                            Discrete(26),  # player3 second last move
                            Discrete(26),  # player3 third last move
                        )
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(26,), dtype=np.int8
                    ),
                }
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {agent: Discrete(26) for agent in self.possible_agents}
        self.turn_list = gen_turn_list()
        self.reward_debug = []

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        string = ""
        for agent in self.agents:
            string += "{}: {}, coins {}, cards {}, rewards {}, \n".format(
                agent,
                MOVES[self.state[agent]["MOVES"][-1]],
                self.state[agent]["COINS"],
                self.state[agent]["CARDS"],
                self.rewards[agent],
            )
        string += str(self.reward_debug) + "\n"
        print(string)

    def reset(self, seed=None, options=None):
        reset_deck()
        self.turn_list = gen_turn_list()
        self.agents = self.possible_agents
        self.rewards = {}
        self.state = {
            agent: {
                "COINS": 0,
                "MOVES": [16, 16, 16],
                "CARDS": [take_card(), take_card(), "None", "None"],
                "TURN": 0,
            }
            for agent in self.agents
        }
        observations = {
            agent: {
                "observation": (
                    0,  # turn
                    0,  # player0 coins
                    0,  # player1 coins
                    0,  # player2 coins
                    0,  # player3 coins
                    CARDS.index(self.state[agent]["CARDS"][0]),  # card 1
                    CARDS.index(self.state[agent]["CARDS"][1]),  # card 2
                    5,  # card 3
                    5,  # card 4
                    16,  # player0 last move
                    16,  # player0 second last move
                    16,  # player0 third last move
                    16,  # player1 last move
                    16,  # player1 second last move
                    16,  # player1 third last move
                    16,  # player2 last move
                    16,  # player2 second last move
                    16,  # player2 third last move
                    16,  # player3 last move
                    16,  # player3 second last move
                    16,  # player3 third last move
                ),
                "action_mask": none_mask,
            }
            for agent in self.agents
        }
        observations["player_0"]["action_mask"] = np.pad(np.array([1, 1, 1]), (12, 11))
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def has_card(self, card, player):
        """
        Returns true if player holds card, else false
        """
        return card in self.state[player]["CARDS"]

    def resolve_challenge(self, card, agent, target, steps):
        """
        Reward players if they pass/fail challenge
        """
        if self.has_card(card, target):
            if card == "ASSASSIN":
                self.state[agent]["CARDS"] = ["None", "None", "None", "None"]
                self.rewards[agent] -= 10
                self.reward_debug.append(
                    f"punish {agent} loses game for failed assasination challenge"
                )
                self.rewards[target] += 10
            else:
                self.rewards[target] += 5
                self.rewards[agent] -= 5
                self.reward_debug.append(f"reward {target} win challenge")
                self.reward_debug.append(f"punish {agent} lose challenge")
                discard_turn = TURNS.index(f"discard-{agent}")
                self.turn_list.insert(steps, discard_turn)
            # swap out revealed card
            card_index = self.state[target]["CARDS"].index(card)
            self.state[target]["CARDS"][card_index] = take_card()
            DECK.insert(0, card)
        else:
            self.rewards[target] -= 5
            self.rewards[agent] += 5
            self.reward_debug.append(f"reward {agent} win challenge")
            self.reward_debug.append(f"punish {target} lose challenge")
            discard_turn = TURNS.index(f"discard-{target}")
            self.turn_list.insert(steps, discard_turn)

    def remove_card(self, card, agent):
        """
        If agent doesn't hold card, punish 10.
        If agent holds card, remove card from state.
        """
        if not card in self.state[agent]["CARDS"]:
            self.rewards[agent] -= 10
            self.reward_debug.append(f"punish {agent} remove card not held")
        else:
            card_index = self.state[agent]["CARDS"].index(card)
            self.state[agent]["CARDS"][card_index] = "None"
            DECK.insert(0, card)

    def last_turn(self, player):
        """
        Return a player's last turn-based move, discounting counters, discards, challenges and None.
        If no moves, return None.
        """
        moves = self.state[player]["MOVES"].copy().reverse()
        if not moves:
            return 25
        for move in moves:
            if move < 16:
                return move
        return 25

    def last_move(self, player):
        """
        Return a player's last action, discounting None.
        If no moves, return None.
        """
        moves = self.state[player]["MOVES"].copy().reverse()
        if not moves:
            return 25
        for move in moves:
            if move != 25:
                return move
        return 25

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

    def get_counters(self, actions):
        counters = []
        for agent, action in actions.items():
            if action in [16, 17, 18]:
                counters.append({"agent": agent, "move": action})
        return counters

    def step(self, actions):
        observations = {}
        infos = {}
        self.reward_debug = []
        self.rewards = {agent: 0 for agent in self.agents}

        turn_step = self.state["player_0"]["TURN"]
        turn = self.turn_list[turn_step]
        leader = TURNS[turn] if turn % 4 == 0 else TURNS[turn].split("-")[-1]
        last_leader_move = self.state[leader]["MOVES"][-(1 + turn % 4)]
        victim = f"player_{last_leader_move % 4}"

        strip_none_actions = [
            [agent, action] for agent, action in actions.items() if action != 25
        ]

        if not len(strip_none_actions):
            strip_none_actions = [["player_0", 25]]

        agent, action = random.choice(strip_none_actions)

        target = "player_" + str(action % 4) if (action < 13) else agent
        target_discard_turn = TURNS.index(f"discard-{target}")

        # punish lack of discard
        if (turn % 4 == 3) and (agent == leader) and not action in [20, 21, 22, 23, 24]:
            self.rewards[agent] -= 10
            self.reward_debug.append(f"punish {agent} not discarding")

        # ignore none
        elif action == 25:
            pass

        # punish incorrect primary turn
        elif (turn % 4 == 0) and (agent != leader):
            self.rewards[agent] -= 10
            self.reward_debug.append(f"punish {agent} wrong turn")

        # punish incorrect counter
        elif (turn % 4 == 1) and not action in [16, 17, 18, 19]:
            self.rewards[agent] -= 10
            self.reward_debug.append(f"punish {agent} bad counter")

        # punish incorrect challenge
        elif (turn % 4 == 2) and action != 19:
            self.rewards[agent] -= 10
            self.reward_debug.append(f"punish {agent} bad challenge")

        # punish lack of wait for discard
        elif (turn % 4 == 3) and (agent != leader):
            self.rewards[agent] -= 10
            self.reward_debug.append(f"punish {agent} not waiting for discard")

        else:
            match action:
                case [0, 1, 2, 3]:  # COUP
                    if self.state[agent]["COINS"] >= 7:
                        self.state[agent]["COINS"] -= 7
                        self.turn_list.insert(turn_step + 3, target_discard_turn)
                        self.rewards[agent] += 5
                        self.rewards[target] -= 5
                        self.reward_debug.append(f"punish {target} coup")
                        self.reward_debug.append(f"reward {agent} coup")
                    else:
                        self.rewards[agent] -= 10
                        self.reward_debug.append(f"punish {agent} cannot afford coup")
                case [4, 5, 6, 7]:  # ASSASSINATE
                    if self.state[agent]["COINS"] >= 3:
                        self.state[agent]["COINS"] -= 3
                        self.turn_list.insert(turn_step + 3, target_discard_turn)
                        self.rewards[agent] += 5
                        self.rewards[target] -= 5
                        self.reward_debug.append(f"reward {agent} assassinate")
                        self.reward_debug.append(f"punish {target} assassinated")
                    else:
                        self.rewards[agent] -= 10
                        self.reward_debug.append(
                            f"punish {agent} cannot afford assassination"
                        )
                case [8, 9, 10, 11]:  # STEAL
                    if self.state[target]["COINS"] >= 2:
                        self.state[agent]["COINS"] += 2
                        self.state[target]["COINS"] -= 2
                        self.rewards[agent] += 2
                        self.rewards[target] -= 2
                        self.reward_debug.append(f"reward {agent} steal 2")
                        self.reward_debug.append(f"punish {target} 2 stolen")
                    elif self.state[target]["COINS"] == 1:
                        self.state[agent]["COINS"] += 1
                        self.state[target]["COINS"] -= 1
                        self.rewards[agent] += 1
                        self.rewards[target] -= 1
                        self.reward_debug.append(f"reward {agent} steal 1")
                        self.reward_debug.append(f"punish {target} 1 stolen")
                case 12:  # INCOME
                    self.state[agent]["COINS"] += 1
                    self.rewards[agent] += 1
                    self.reward_debug.append(f"reward {agent} income")
                case 13:  # FOREIGN AID
                    self.state[agent]["COINS"] += 2
                    self.rewards[agent] += 2
                    self.reward_debug.append(f"reward {agent} FE")
                case 14:  # TAX
                    self.state[agent]["COINS"] += 3
                    self.rewards[agent] += 3
                    self.reward_debug.append(f"reward {agent} tax")
                case 15:  # EXCHANGE
                    try:
                        cards = self.state[agent]["CARDS"]
                        for n in range(2):
                            free_slot = cards.index("None")
                            self.state[agent]["CARDS"][free_slot] = take_card()
                            self.turn_list.insert(turn_step + 3, target_discard_turn)
                            if self.number_of_cards(agent) < 3:
                                break
                    except ValueError:
                        self.rewards[agent] -= 3
                        self.reward_debug.append(f"punish {agent} invalid exchange")
                case 16:  # BLOCK FOREIGN AID
                    if last_leader_move != 13:
                        self.rewards[agent] -= 10
                        self.reward_debug.append(f"punish {agent} invalid block FE")
                    else:
                        self.state[leader]["COINS"] -= 2
                        self.rewards[leader] -= 2
                        self.rewards[agent] += 2
                        self.reward_debug.append(f"punish {leader} FE blocked")
                        self.reward_debug.append(f"reward {agent} block FE")
                case 17:  # BLOCK STEALING
                    if not last_leader_move in [8, 9, 10, 11]:
                        self.rewards[agent] -= 10
                        self.reward_debug.append(f"punish {agent} invalid block steal")
                    else:
                        self.state[leader]["COINS"] -= 2
                        self.state[victim]["COINS"] += 2
                        self.rewards[leader] -= -2
                        self.rewards[victim] += 2
                        self.reward_debug.append(f"punish {leader} stealing blocked")
                        self.reward_debug.append(
                            f"reward {victim} stolen coins returned"
                        )
                case 18:  # BLOCK ASSASSINATION
                    if not last_leader_move in [4, 5, 6, 7]:
                        self.rewards[agent] -= 10
                        self.reward_debug.append(
                            f"punish {agent} invalid block assassination"
                        )
                    else:
                        self.turn_list.pop(turn_step + 2)
                        self.rewards[leader] -= 5
                        self.rewards[victim] += 5
                        self.reward_debug.append(f"reward {victim} block assassination")
                        self.reward_debug.append(
                            f"punish {leader} assassination blocked"
                        )
                case 19:  # CHALLENGE
                    challenged_move = self.state[agent]["MOVES"][-1]
                    if turn % 4 == 1:
                        challenged_move = last_leader_move
                        target = leader
                        steps = turn_step + 2
                    else:
                        counters = self.get_counters(actions)
                        steps = turn_step + 1
                        if len(counters):
                            counter = random.choice(self.get_counters())
                            challenged_move = counter["move"]
                            target = counter["agent"]
                    if challenged_move in [14, 16]:
                        self.resolve_challenge("DUKE", agent, target, steps)
                    elif challenged_move in [4, 5, 6, 7]:
                        self.resolve_challenge("ASSASSIN", agent, target, steps)
                    elif challenged_move == 15:
                        self.resolve_challenge("AMBASSADOR", agent, target, steps)
                    elif challenged_move in [8, 9, 10, 11]:
                        self.resolve_challenge("CAPTAIN", agent, target, steps)
                    elif challenged_move == 17:
                        if self.has_card("CAPTAIN", target):
                            self.resolve_challenge("CAPTAIN", agent, target, steps)
                        else:
                            self.resolve_challenge("AMBASSADOR", agent, target, steps)
                    elif challenged_move == 18:
                        self.resolve_challenge("CONTESSA", agent, target, steps)
                    else:
                        self.rewards[agent] -= 10
                        self.reward_debug.append(
                            f"punish {agent} challenged generic move"
                        )
                case 20:  # DISCARD DUKE
                    self.remove_card("DUKE", agent)
                case 21:  # DISCARD CONTESSA
                    self.remove_card("CONTESSA", agent)
                case 22:  # DISCARD CAPTAIN
                    self.remove_card("CAPTAIN", agent)
                case 23:  # DISCARD AMBASSADOR
                    self.remove_card("AMBASSADOR", agent)
                case 24:  # DISCARD ASSASSIN
                    self.remove_card("ASSASSIN", agent)
                case 25:  # None
                    pass

        # ensure coins remain positive
        for agent in self.agents:
            if self.state[agent]["COINS"] < 0:
                self.state[agent]["COINS"] = 0

        had_moves = False
        for agent, action in actions.items():
            if action != 25:
                had_moves = True
            self.state[agent]["MOVES"].append(action)

        next_turn = self.turn_list[turn_step + 1]

        for agent in self.agents:
            self.state[agent]["TURN"] += 1
            action_mask = none_mask
            if (next_turn % 4 == 0) and agent == TURNS[next_turn]:
                action_mask = leader_mask
                # need masks for various money things
                # mask for assassinate and mask for coup
                # mask for steal if others have money...

            elif next_turn % 4 == 1:
                leader = TURNS[next_turn].split("-")[-1]
                last_leader_move = self.state[leader]["MOVES"][-2]
                if agent == leader:
                    pass
                elif last_leader_move == 13:
                    action_mask = counter_fe_mask
                elif last_leader_move in [8, 9, 10, 11]:
                    action_mask = counter_stealing_mask
                elif last_leader_move in [4, 5, 6, 7]:
                    action_mask = counter_assassin_mask
            elif next_turn % 4 == 2:
                if self.state[agent]["MOVES"][-2] == 25 and had_moves:
                    action_mask = challenge_mask
            elif next_turn % 4 == 3:
                if agent == TURNS[next_turn].split("-")[-1]:
                    cards = self.state[agent]["CARDS"]
                    cards = [CARDS.index(card) + 20 for card in cards if card != "None"]
                    zeros = [0] * 26
                    for card in cards:
                        zeros[card] = 1
                    action_mask = np.array(zeros)

            observations[agent] = {
                "observation": (
                    turn_step + 1,
                    self.state["player_0"]["COINS"],
                    self.state["player_1"]["COINS"],
                    self.state["player_2"]["COINS"],
                    self.state["player_3"]["COINS"],
                    CARDS.index(self.state[agent]["CARDS"][0]),
                    CARDS.index(self.state[agent]["CARDS"][1]),
                    CARDS.index(self.state[agent]["CARDS"][2]),
                    CARDS.index(self.state[agent]["CARDS"][3]),
                    self.state["player_0"]["MOVES"][-1],
                    self.state["player_0"]["MOVES"][-2],
                    self.state["player_0"]["MOVES"][-3],
                    self.state["player_1"]["MOVES"][-1],
                    self.state["player_1"]["MOVES"][-2],
                    self.state["player_1"]["MOVES"][-3],
                    self.state["player_2"]["MOVES"][-1],
                    self.state["player_2"]["MOVES"][-2],
                    self.state["player_2"]["MOVES"][-3],
                    self.state["player_3"]["MOVES"][-1],
                    self.state["player_3"]["MOVES"][-2],
                    self.state["player_3"]["MOVES"][-3],
                ),
                "action_mask": action_mask,
            }
            infos[agent] = {}

        env_truncation = self.state["player_0"]["TURN"] >= NUM_ITERS

        terminations = {
            agent: self.number_of_cards(agent) == 0 for agent in self.agents
        }
        truncations = {agent: env_truncation for agent in self.agents}

        players_left = [item[0] for item in terminations.items() if item[1] == False]

        if self.render_mode in ["human", "all"]:
            self.render()

        if env_truncation or len(players_left) == 1:
            print("Game Over")
            self.rewards[players_left[0]] += 30
            self.agents = []

        return observations, self.rewards, terminations, truncations, infos
