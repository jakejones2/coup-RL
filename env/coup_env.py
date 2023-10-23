import functools
import random

import numpy as np
from gymnasium.spaces import Discrete, Box
from gymnasium import spaces

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


MOVES = [
    "STEAL0",  # 0
    "STEAL1",  # 1
    "STEAL2",  # 2
    "STEAL3",  # 3
    "COUP0",  # 4
    "COUP1",  # 5
    "COUP2",  # 6
    "COUP3",  # 7
    "ASSASSINATE0",  # 8
    "ASSASSINATE1",  # 9
    "ASSASSINATE2",  # 10
    "ASSASSINATE3",  # 11
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
    "",  # 5
]

TURNS = [
    "player_0",
    "counter-player_0",
    "challenge-counter-player_0",
    "discard-player_0",
    "player_1",
    "counter-player_1",
    "challenge-counter-player_1",
    "discard-player_1",
    "player_2",
    "counter-player_2",
    "challenge-counter-player_2",
    "discard-player_2",
    "player_3",
    "counter-player_3",
    "challenge-counter-player_3",
    "discard-player_3",
]

NUM_ITERS = 80

leader_mask = np.pad(np.ones(4), (12, 10))
three_coin_mask = np.pad(np.ones(8), (8, 10))
seven_coin_mask = np.pad(np.ones(12), (4, 10))
ten_card_mask = np.pad(np.ones(4), (4, 18))
none_mask = np.append(np.zeros(25), [1])
challenge_mask = np.append(np.pad(np.array([1]), (19, 5)), [1])
counter_fe_mask = np.append(np.pad(np.array([1]), (16, 8)), [1])
counter_stealing_mask = np.append(np.pad(np.array([1, 0, 1]), (17, 5)), [1])
counter_assassin_mask = np.append(np.pad(np.array([1, 1]), (18, 5)), [1])


def gen_turn_list():
    round = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    rounds = []
    for n in range(50):
        rounds.extend(round)
    return rounds


def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = CoupFourPlayers(render_mode=render_mode)
    # env = parallel_to_aec(env)
    return env


class Deck:
    def __init__(self, size=3):
        self.size = size
        self.deck = ["ASSASSIN", "AMBASSADOR", "DUKE", "CONTESSA", "CAPTAIN"] * size
        random.shuffle(self.deck)

    def take(self):
        return self.deck.pop()

    def add(self, card):
        self.deck.insert(0, card)

    def reset(self):
        self.deck = [
            "ASSASSIN",
            "AMBASSADOR",
            "DUKE",
            "CONTESSA",
            "CAPTAIN",
        ] * self.size
        random.shuffle(self.deck)


class CoupFourPlayers(ParallelEnv):
    """
    Environment for the card game 'Coup'. See rules here: https://www.ultraboardgames.com/coup/game-rules.php
    """

    metadata = {"render_modes": ["human"], "name": "coup_v0"}

    def __init__(self, render_mode=None, deck=Deck(3)):
        # need to make number of players dynamic
        self.agents = ["player_" + str(r) for r in range(4)]
        self.deck = deck
        self.render_mode = render_mode
        self.rewards = {}
        self.turn_list = gen_turn_list()
        self.reward_msgs = []
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observations": spaces.Box(
                        low=0, high=300, shape=(21,), dtype=np.float32
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(26,), dtype=np.float32
                    ),
                }
            )
            for agent in self.agents
        }
        self.action_spaces = {agent: Discrete(26) for agent in self.agents}

    # @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        next_turn = self.turn_list[self.state[agent]["TURN"]]
        coins = self.state[agent]["COINS"]

        # determine player data
        stealable = []
        dead = []
        action = 25
        for pos_agent in self.agents:
            last_move = self.state[pos_agent]["MOVES"][-1]
            if last_move != 25:
                action = last_move
            if coins > 0 and self.number_of_cards(pos_agent):
                stealable.append(int(pos_agent[-1]))
            if self.number_of_cards(pos_agent) == 0:
                dead.append(int(pos_agent[-1]) + 4)
                dead.append(int(pos_agent[-1]) + 8)

        action_mask = none_mask
        if (next_turn % 4 == 0) and (agent == TURNS[next_turn]):
            action_mask = leader_mask
            if coins >= 3:
                action_mask = three_coin_mask
            if coins >= 7:
                action_mask = seven_coin_mask
            # can steal from those with coins
            for action in stealable:
                action_mask[action] = 1
            # cannot coup or assassinate dead players
            for action in dead:
                action_mask[action] = 0
            if coins >= 10:
                action = ten_card_mask
        elif next_turn % 4 == 1:
            leader = TURNS[next_turn].split("-")[-1]
            last_leader_move = self.state[leader]["MOVES"][-1]
            if agent == leader:
                pass
            elif last_leader_move == 13:
                action_mask = counter_fe_mask
            elif last_leader_move < 4:
                action_mask = counter_stealing_mask
            elif last_leader_move in [8, 9, 10, 11]:
                action_mask = counter_assassin_mask
            elif last_leader_move in [14, 15]:
                action_mask = challenge_mask
        elif next_turn % 4 == 2:
            if self.state[agent]["MOVES"][-1] == 25 and not action in [25, 19]:
                action_mask = challenge_mask
        elif next_turn % 4 == 3:
            if agent == TURNS[next_turn].split("-")[-1]:
                cards = self.state[agent]["CARDS"]
                cards = [CARDS.index(card) + 20 for card in cards if card]
                zeros = [0] * 26
                for card in cards:
                    zeros[card] = 1
                action_mask = np.array(zeros)

        # ensure cannot act against self
        agent_num = int(agent[-1])
        action_mask[agent_num] = 0
        action_mask[agent_num + 4] = 0
        action_mask[agent_num + 8] = 0

        # ensure dead players cannot act
        if self.number_of_cards(agent) == 0:
            action_mask = none_mask

        return {
            "observations": np.array(
                [
                    next_turn,
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
                ],
                dtype=np.float32,
            ),
            "action_mask": action_mask.astype(np.int8),
        }

    def render(self, last_turn):
        string = f"\n----{TURNS[last_turn]}----\n"
        for agent in self.agents:
            string += "{}: {}, coins {}, cards {}, rewards {}, \n".format(
                agent,
                MOVES[self.state[agent]["MOVES"][-1]],
                self.state[agent]["COINS"],
                [card for card in self.state[agent]["CARDS"] if card],
                self.rewards[agent],
            )
        if len(self.reward_msgs):
            string += str(self.reward_msgs)
        print(string)

    def reset(self, seed=None, options=None):
        self.deck.reset()
        self.turn_list = gen_turn_list()
        self.rewards = {}
        self.state = {
            agent: {
                "COINS": 0,
                "MOVES": [25, 25, 25],
                "CARDS": [self.deck.take(), self.deck.take(), "", ""],
                "TURN": 0,
            }
            for agent in self.agents
        }
        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def has_card(self, card, player):
        """
        Returns true if player holds card, else false
        """
        return card in self.state[player]["CARDS"][:-2]

    def resolve_challenge(self, card, agent, target, steps):
        """
        Reward players if they pass/fail challenge
        """
        if self.has_card(card, target):
            if card == "ASSASSIN":
                self.state[agent]["CARDS"] = ["", "", "", ""]
                self.rewards[agent] -= 10
                self.reward_msgs.append(
                    f"punish {agent} loses game for failed assasination challenge"
                )
                self.rewards[target] += 10
            else:
                self.rewards[target] += 5
                self.rewards[agent] -= 5
                self.reward_msgs.append(f"reward {target} win challenge")
                self.reward_msgs.append(f"punish {agent} lose challenge")
                discard_turn = TURNS.index(f"discard-{agent}")
                self.turn_list.insert(steps, discard_turn)
            # swap out revealed card
            card_index = self.state[target]["CARDS"].index(card)
            self.state[target]["CARDS"][card_index] = self.deck.take()
            self.deck.add(card)
            return False
        else:
            self.rewards[target] -= 5
            self.rewards[agent] += 5
            self.reward_msgs.append(f"reward {agent} win challenge")
            self.reward_msgs.append(f"punish {target} lose challenge")
            discard_turn = TURNS.index(f"discard-{target}")
            self.turn_list.insert(steps, discard_turn)
            return True

    def remove_card(self, card, agent):
        """
        If agent doesn't hold card, punish 10.
        If agent holds card, remove card from state.
        """
        if not card in self.state[agent]["CARDS"]:
            self.rewards[agent] -= 10
            self.reward_msgs.append(f"punish {agent} remove card not held")
        else:
            card_index = self.state[agent]["CARDS"].index(card)
            self.state[agent]["CARDS"][card_index] = ""
            self.deck.add(card)
            self.state[agent]["CARDS"].sort(reverse=True)

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
            if card:
                count += 1
        return count

    def get_counters(self):
        counters = []
        for agent in self.agents:
            last_move = self.state[agent]["MOVES"][-1]
            if last_move in [16, 17, 18]:
                counters.append({"agent": agent, "move": last_move})
        return counters

    def step(self, actions):
        observations = {}
        infos = {}
        self.reward_msgs = []
        self.rewards = {agent: 0 for agent in self.agents}

        turn_step = self.state["player_0"]["TURN"]
        turn = self.turn_list[turn_step]
        leader = TURNS[turn] if turn % 4 == 0 else TURNS[turn].split("-")[-1]
        last_leader_move = self.state[leader]["MOVES"][-(turn % 4)]
        victim = f"player_{last_leader_move % 4}"

        actions = [[agent, action] for agent, action in actions.items() if action != 25]

        if not len(actions):
            actions = [["player_0", 25]]

        actor, action = random.choice(actions)

        target = "player_" + str(action % 4) if (action < 12) else actor
        target_discard_turn = TURNS.index(f"discard-{target}")

        # punish lack of discard
        if (turn % 4 == 3) and (actor == leader) and not action in [20, 21, 22, 23, 24]:
            self.rewards[actor] -= 10
            self.reward_msgs.append(f"punish {actor} not discarding")

        # ignore none
        elif action == 25:
            pass

        # punish incorrect primary turn
        elif (turn % 4 == 0) and (actor != leader):
            self.rewards[actor] -= 10
            self.reward_msgs.append(f"punish {actor} wrong turn")

        # punish incorrect counter
        elif (turn % 4 == 1) and not action in [16, 17, 18, 19]:
            self.rewards[actor] -= 10
            self.reward_msgs.append(f"punish {actor} bad counter")

        # punish incorrect challenge
        elif (turn % 4 == 2) and action != 19:
            self.rewards[actor] -= 10
            self.reward_msgs.append(f"punish {actor} bad challenge")

        # punish lack of wait for discard
        elif (turn % 4 == 3) and (actor != leader):
            self.rewards[actor] -= 10
            self.reward_msgs.append(f"punish {actor} not waiting for discard")

        else:
            match action:
                case 0 | 1 | 2 | 3:  # STEAL
                    if self.state[target]["COINS"] >= 2:
                        self.state[actor]["COINS"] += 2
                        self.state[target]["COINS"] -= 2
                        self.rewards[actor] += 2
                        self.rewards[target] -= 2
                        self.reward_msgs.append(f"reward {actor} steal 2")
                        self.reward_msgs.append(f"punish {target} 2 stolen")
                    elif self.state[target]["COINS"] == 1:
                        self.state[actor]["COINS"] += 1
                        self.state[target]["COINS"] -= 1
                        self.rewards[actor] += 1
                        self.rewards[target] -= 1
                        self.reward_msgs.append(f"reward {actor} steal 1")
                        self.reward_msgs.append(f"punish {target} 1 stolen")
                case 4 | 5 | 6 | 7:  # COUP
                    if self.state[actor]["COINS"] >= 7:
                        self.state[actor]["COINS"] -= 7
                        self.turn_list.insert(turn_step + 3, target_discard_turn)
                        self.rewards[actor] += 5
                        self.rewards[target] -= 5
                        self.reward_msgs.append(f"punish {target} coup")
                        self.reward_msgs.append(f"reward {actor} coup")
                    else:
                        self.rewards[actor] -= 10
                        self.reward_msgs.append(f"punish {actor} cannot afford coup")
                case 8 | 9 | 10 | 11:  # ASSASSINATE
                    if self.state[actor]["COINS"] >= 3:
                        self.state[actor]["COINS"] -= 3
                        self.turn_list.insert(turn_step + 3, target_discard_turn)
                        self.rewards[actor] += 5
                        self.rewards[target] -= 5
                        self.reward_msgs.append(f"reward {actor} assassinate")
                        self.reward_msgs.append(f"punish {target} assassinated")
                    else:
                        self.rewards[actor] -= 10
                        self.reward_msgs.append(
                            f"punish {actor} cannot afford assassination"
                        )
                case 12:  # INCOME
                    self.state[actor]["COINS"] += 1
                    self.rewards[actor] += 1
                    self.reward_msgs.append(f"reward {actor} income")
                case 13:  # FOREIGN AID
                    self.state[actor]["COINS"] += 2
                    self.rewards[actor] += 2
                    self.reward_msgs.append(f"reward {actor} FE")
                case 14:  # TAX
                    self.state[actor]["COINS"] += 3
                    self.rewards[actor] += 3
                    self.reward_msgs.append(f"reward {actor} tax")
                case 15:  # EXCHANGE
                    try:
                        cards = self.state[actor]["CARDS"]
                        for n in range(2):
                            free_slot = cards.index("")
                            self.state[actor]["CARDS"][free_slot] = self.deck.take()
                            self.turn_list.insert(turn_step + 3, target_discard_turn)
                            if self.number_of_cards(actor) < 3:
                                break
                        self.state[actor]["CARDS"].sort(reverse=True)
                    except ValueError:
                        self.rewards[actor] -= 3
                        self.reward_msgs.append(f"punish {actor} invalid exchange")
                case 16:  # BLOCK FOREIGN AID
                    if last_leader_move != 13:
                        self.rewards[actor] -= 10
                        self.reward_msgs.append(f"punish {actor} invalid block FE")
                    else:
                        self.state[leader]["COINS"] -= 2
                        self.rewards[leader] -= 2
                        self.rewards[actor] += 2
                        self.reward_msgs.append(f"punish {leader} FE blocked")
                        self.reward_msgs.append(f"reward {actor} block FE")
                case 17:  # BLOCK STEALING
                    if not last_leader_move < 4:
                        self.rewards[actor] -= 10
                        self.reward_msgs.append(f"punish {actor} invalid block steal")
                    else:
                        self.state[leader]["COINS"] -= 2
                        self.state[victim]["COINS"] += 2
                        self.rewards[leader] -= -2
                        self.rewards[victim] += 2
                        self.reward_msgs.append(f"punish {leader} stealing blocked")
                        self.reward_msgs.append(
                            f"reward {victim} stolen coins returned"
                        )
                case 18:  # BLOCK ASSASSINATION
                    if not last_leader_move in [8, 9, 10, 11]:
                        self.rewards[actor] -= 10
                        self.reward_msgs.append(
                            f"punish {actor} invalid block assassination"
                        )
                    else:
                        self.turn_list.pop(turn_step + 2)
                        self.rewards[leader] -= 5
                        self.rewards[victim] += 5
                        self.reward_msgs.append(f"reward {victim} avoids assassination")
                        self.reward_msgs.append(
                            f"punish {leader} assassination blocked"
                        )
                case 19:  # CHALLENGE
                    # prepare
                    if turn % 4 == 1:
                        challenged_move = last_leader_move
                        target = leader
                        steps = turn_step + 2
                    else:
                        counters = self.get_counters()
                        steps = turn_step + 1
                        if len(counters):
                            counter = random.choice(self.get_counters())
                            challenged_move = counter["move"]
                            target = counter["agent"]
                        else:
                            challenged_move = 25
                    # execute
                    if challenged_move == 25:
                        pass
                    elif challenged_move in [14, 16]:
                        fail = self.resolve_challenge("DUKE", actor, target, steps)
                        if fail and challenged_move == 14:
                            self.state[target]["COINS"] -= 3
                            self.rewards[target] -= 3
                            self.reward_msgs.append(
                                f"punish {target} loses former tax reward"
                            )
                        if fail and challenged_move == 16:
                            self.state[leader]["COINS"] += 2
                            self.rewards[leader] += 2
                            self.rewards[target] -= 2
                            self.reward_msgs.append(f"reward {leader} regains FE")
                            self.reward_msgs.append(
                                f"punish {target} loses former block FE reward"
                            )
                    elif challenged_move in [8, 9, 10, 11]:
                        fail = self.resolve_challenge("ASSASSIN", actor, target, steps)
                        if fail:
                            self.rewards[target] -= 5
                            self.reward_msgs.append(
                                f"punish {target} loses former assassination reward"
                            )
                        if not fail:
                            self.turn_list.pop(turn_step + 2)
                            self.rewards[victim] += 5
                            self.rewards[target] -= 5
                            self.reward_msgs.append(
                                f"reward {victim} avoids assassination"
                            )
                            self.reward_msgs.append(
                                f"punish {target} loses assassination reward"
                            )

                    elif challenged_move == 15:
                        fail = self.resolve_challenge(
                            "AMBASSADOR", actor, target, steps
                        )
                        if fail:
                            self.state[target]["CARDS"][2] = ""
                            self.state[target]["CARDS"][3] = ""
                            self.turn_list.pop(turn_step + 2)
                            self.turn_list.pop(turn_step + 2)

                    elif challenged_move < 4:
                        fail = self.resolve_challenge("CAPTAIN", actor, target, steps)
                        if fail:
                            self.state[victim]["COINS"] += 2
                            self.state[target]["COINS"] -= 2
                            self.rewards[victim] += 2
                            self.rewards[target] -= 2
                            self.reward_msgs.append(
                                f"reward {victim} wins back stolen coins"
                            )
                            self.reward_msgs.append(
                                f"punish {target} loses former steal reward"
                            )
                    elif challenged_move == 17:
                        if self.has_card("CAPTAIN", target):
                            fail = self.resolve_challenge(
                                "CAPTAIN", actor, target, steps
                            )
                        else:
                            fail = self.resolve_challenge(
                                "AMBASSADOR", actor, target, steps
                            )
                        if fail:
                            self.state[leader]["COINS"] += 2
                            self.state[victim]["COINS"] -= 2
                            self.rewards[leader] += 2
                            self.rewards[victim] -= 2
                            self.reward_msgs.append(
                                f"punish {victim} loses coins despite block"
                            )
                            self.reward_msgs.append(
                                f"reward {leader} regains stolen coins"
                            )
                    elif challenged_move == 18:
                        self.resolve_challenge("CONTESSA", actor, target, steps)
                        # failure handled within resolution
                    else:
                        self.rewards[actor] -= 10
                        self.reward_msgs.append(
                            f"punish {actor} challenged generic move"
                        )
                case 20:  # DISCARD DUKE
                    self.remove_card("DUKE", actor)
                case 21:  # DISCARD CONTESSA
                    self.remove_card("CONTESSA", actor)
                case 22:  # DISCARD CAPTAIN
                    self.remove_card("CAPTAIN", actor)
                case 23:  # DISCARD AMBASSADOR
                    self.remove_card("AMBASSADOR", actor)
                case 24:  # DISCARD ASSASSIN
                    self.remove_card("ASSASSIN", actor)
                case 25:  # None
                    pass

        # determine next turn
        new_step = turn_step + 1
        if action in [19, 25] and turn % 4 == 1:
            new_step += 1
        elif action in [4, 5, 6, 7, 12, 25] and turn % 4 == 0:
            new_step += 2

        # update all state
        self.state[actor]["MOVES"].append(action)
        for agent in self.agents:
            self.state[agent]["TURN"] = new_step
            coins = self.state[agent]["COINS"]
            if coins <= 0:
                self.state[agent]["COINS"] = 0
            if agent != actor:
                self.state[agent]["MOVES"].append(25)

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        env_truncation = new_step >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}
        # seems that you cannot report one agent terminated with RLlib?
        # this is potentially the source of the single trajectory error
        # https://github.com/ray-project/ray/issues/10761
        dead = {agent: self.number_of_cards(agent) == 0 for agent in self.agents}
        players_left = [item[0] for item in dead.items() if item[1] == False]
        terminations = {agent: False for agent in self.agents}

        if self.render_mode in ["human"]:
            self.render(turn)

        if env_truncation or len(players_left) == 1:
            print(f"Game Over - {players_left[0]} wins!")
            terminations["__all__"] = True
            truncations["__all__"] = True
            self.rewards[players_left[0]] += 30
            # self.agents = []

        return observations, self.rewards, terminations, truncations, infos
