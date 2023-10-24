"""
PettingZoo (multiagent Gymnasium) reinforcement learning environment for the card 
game 'Coup'. https://www.ultraboardgames.com/coup/game-rules.php

This is a parallel environment. Import the class CoupFourPlayers directly, 
or the env function which adds PettingZoo wrappers and the option to convert to the 
AEC (turn-based) API.

This is a work in progress. It's possible that the current reward and action-masking
system isn't conducive to good learning with standard algorithms - the best policy I have
achieved so far is 30% better than average, winning the four-player game 55% of the time 
(n=4000).

Rewards could be simplified as proportional to coins and inversely proportional to the 
number of players remaining, with perhaps a winning bonus. Environment could also be 
modified to take a variable number of agents rather than just 4.

For a quick implementation, run the file random_policy.py. Environment render_mode parameter
can be "moves" for a full breakdown or "games" for seeing summary. Anything else (e.g. "none") 
will render nothing.
"""

import functools
import random
import math
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict
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
    "player_0",  # 0
    "counter-player_0",  # 1
    "challenge-counter-player_0",  # 2
    "discard-player_0",  # 3
    "player_1",  # 4
    "counter-player_1",  # 5
    "challenge-counter-player_1",  # 6
    "discard-player_1",  # 7
    "player_2",  # 8
    "counter-player_2",  # 9
    "challenge-counter-player_2",  # 10
    "discard-player_2",  # 11
    "player_3",  # 12
    "counter-player_3",  # 13
    "challenge-counter-player_3",  # 14
    "discard-player_3",  # 15
]

# number of iterations before env truncation
NUM_ITERS = 80

# action mask templates, each must be an array of length 26 containing 1s and 0s.
LEADER_MASK = np.pad(np.ones(4), (12, 10))
THREE_COIN_MASK = np.pad(np.ones(8), (8, 10))
SEVEN_COIN_MASK = np.pad(np.ones(12), (4, 10))
TEN_CARD_MASK = np.pad(np.ones(4), (4, 18))
NONE_MASK = np.append(np.zeros(25), [1])
CHALLENGE_MASK = np.append(np.pad(np.array([1]), (19, 5)), [1])
COUNTER_FE_MASK = np.append(np.pad(np.array([1]), (16, 8)), [1])
COUNTER_STEALING_MASK = np.append(np.pad(np.array([1, 0, 1]), (17, 5)), [1])
COUNTER_ASSASSINATION_MASK = np.append(np.pad(np.array([1, 1]), (18, 5)), [1])


def gen_turn_list():
    """
    Generates a starting turn list for the environment. Indexes the "TURNS" variable
    (see top of file).

    For each turn within cycle:
        If turn % 4 == 0, turn is of type primary move.
        If turn % 4 == 1, turn is of type counter.
        If turn % 4 == 2, turn is of type challenge-counter.
        If turn % 4 == 3, turn is of type discard.

    Each cycle thus follows the following formula:
        player_0 (makes primary move, e.g foreign aid)
        counter-player_0 (either challenges or blocks primary move)
        challenge-counter-player_0 (challenges counter if applicable)
        player_1
        counter-player_1
        etc.

    Discard rounds are added into the turn list as needed by the step
    function. The step function also skips rounds when not legal (e.g.
    counter-player_N is skipped if action is not counterable, and
    challenge-counter-player_N is skipped if there is no counter).
    """
    cycle = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    turn_list = []
    for n in range(math.ceil(NUM_ITERS / 4)):
        turn_list.extend(cycle)
    return turn_list


def env(render_mode=None, convert_to_aec=False):
    """
    Use this function to add PettingZoo wrappers to env.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "moves"
    env = CoupFourPlayers(render_mode=internal_render_mode)
    if convert_to_aec:
        env = parallel_to_aec(env)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class Deck:
    """
    Class to model the deck of cards in play.
    """

    def __init__(self, size=3):
        self.size = size
        self.deck = ["ASSASSIN", "AMBASSADOR", "DUKE", "CONTESSA", "CAPTAIN"] * size
        random.shuffle(self.deck)

    def take(self):
        try:
            return self.deck.pop()
        except IndexError:
            return CARDS[random.randint(0, 4)]

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
    Four-player environment for the card game 'Coup'.
    See rules here: https://www.ultraboardgames.com/coup/game-rules.php
    """

    metadata = {"render_modes": ["moves, games, ansi"], "name": "coup_v0"}

    def __init__(self, render_mode=None, deck=Deck(3)):
        self.agents = ["player_" + str(r) for r in range(4)]
        self.deck = deck
        self.render_mode = render_mode
        self.rewards = {}
        self.cumulative_rewards = {agent: 0 for agent in self.agents}
        self.turn_list = gen_turn_list()
        self.reward_msgs = []
        self.observation_spaces = {
            agent: Dict(
                {
                    "observations": Box(low=0, high=300, shape=(21,), dtype=np.float32),
                    "action_mask": Box(low=0, high=1, shape=(26,), dtype=np.int8),
                }
            )
            for agent in self.agents
        }
        self.action_spaces = {agent: Discrete(26) for agent in self.agents}
        self.game_ended = False

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Returns shape of the observations space for given agent.
        """
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Returns shape of the action space for given agent.
        """
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        """
        Resets environemnt state for a new game, and returns initial
        observations and info.
        """
        self.game_ended = False
        self.deck.reset()
        self.turn_list = gen_turn_list()
        self.rewards = {}
        self.cumulative_rewards = {agent: 0 for agent in self.agents}
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

    def observe(self, agent):
        """
        Creates observation for given player.
        Observation is a dict of 'action_mask' and 'observations'.
        """
        next_turn = self.turn_list[self.state[agent]["TURN"]]
        coins = self.state[agent]["COINS"]
        agent_num = int(agent[-1])

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

        action_mask = NONE_MASK
        if (next_turn % 4 == 0) and (agent == TURNS[next_turn]):
            action_mask = LEADER_MASK
            if coins >= 3:
                action_mask = THREE_COIN_MASK
            if coins >= 7:
                action_mask = SEVEN_COIN_MASK
            # can steal from those with coins
            for action in stealable:
                action_mask[action] = 1
            # cannot coup or assassinate dead players
            for action in dead:
                action_mask[action] = 0
            if coins >= 10:
                action = TEN_CARD_MASK
        elif next_turn % 4 == 1:
            leader = TURNS[next_turn].split("-")[-1]
            last_leader_move = self.state[leader]["MOVES"][-1]
            if agent == leader:
                pass
            elif last_leader_move == 13:
                action_mask = COUNTER_FE_MASK
            elif last_leader_move < 4:
                action_mask = COUNTER_STEALING_MASK
            elif last_leader_move in [8, 9, 10, 11]:
                if last_leader_move % 4 == agent_num:
                    action_mask = COUNTER_ASSASSINATION_MASK
                else:
                    action_mask = CHALLENGE_MASK
            elif last_leader_move in [14, 15]:
                action_mask = CHALLENGE_MASK
        elif next_turn % 4 == 2:
            if self.state[agent]["MOVES"][-1] == 25 and not action in [25, 19]:
                action_mask = CHALLENGE_MASK
        elif next_turn % 4 == 3:
            if agent == TURNS[next_turn].split("-")[-1]:
                cards = self.state[agent]["CARDS"]
                cards = [CARDS.index(card) + 20 for card in cards if card]
                zeros = [0] * 26
                for card in cards:
                    zeros[card] = 1
                action_mask = np.array(zeros)

        # ensure cannot act against self
        action_mask[agent_num] = 0
        action_mask[agent_num + 4] = 0
        action_mask[agent_num + 8] = 0

        # ensure dead players cannot act
        if self.number_of_cards(agent) == 0:
            action_mask = NONE_MASK

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

    def step(self, actions):
        """
        This function handles game actions by updating environment state and allocating rewards.
        Each step, every agent submits an action which indexes a move. Moves are listed as
        MOVES at the top of the file, and include a 'None' move (25).

        The game has a predetermined sequence of turns (self.turn_list). Each player has a corresponding
        turn state (e.g. self.state["player_0"]["TURN"]) which indexes self.turn_list. Each player's
        turn state is identical, and increments every step by a variable amount depending on play, thus
        progressing through self.turn_list with the option to skip turns. The turn sequence (self.turn_list)
        can also be modified, e.g. when an action results in a player needing to discard, a discard turn
        is inserted into the sequence and implemented when the "TURN" state increments to this index.
        The turn sequence (self.turn_list) itself indexes possible turns listed at the top of the
        file in the variable "TURNS".
        """
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
                    if last_leader_move != 8 + int(actor[-1]):
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
                        next_discard_step = turn_step + 2
                    else:
                        counters = self.get_counters()
                        next_discard_step = turn_step + 1
                        if len(counters):
                            counter = random.choice(counters)
                            challenged_move = counter["move"]
                            target = counter["agent"]
                        else:
                            challenged_move = 25
                    # execute
                    if challenged_move == 25:
                        pass
                    elif challenged_move in [14, 16]:
                        win_challenge = self.resolve_challenge(
                            "DUKE", actor, target, next_discard_step
                        )
                        if win_challenge and challenged_move == 14:
                            self.state[target]["COINS"] -= 3
                            self.rewards[target] -= 3
                            self.reward_msgs.append(
                                f"punish {target} loses former tax reward"
                            )
                        if win_challenge and challenged_move == 16:
                            self.state[leader]["COINS"] += 2
                            self.rewards[leader] += 2
                            self.rewards[target] -= 2
                            self.reward_msgs.append(f"reward {leader} regains FE")
                            self.reward_msgs.append(
                                f"punish {target} loses former block FE reward"
                            )
                    elif challenged_move in [8, 9, 10, 11]:
                        win_challenge = self.resolve_challenge(
                            "ASSASSIN", actor, target, next_discard_step
                        )
                        if win_challenge:
                            self.turn_list.pop(
                                turn_step + 3
                            )  # after the challenge discard
                            self.rewards[victim] += 5
                            self.rewards[target] -= 5
                            self.reward_msgs.append(
                                f"reward {victim} avoids assassination"
                            )
                            self.reward_msgs.append(
                                f"punish {target} loses former assassination reward"
                            )
                    elif challenged_move == 15:
                        win_challenge = self.resolve_challenge(
                            "AMBASSADOR", actor, target, next_discard_step
                        )
                        if win_challenge:
                            self.state[target]["CARDS"][2] = ""
                            self.state[target]["CARDS"][3] = ""
                            self.turn_list.pop(turn_step + 3)
                            self.turn_list.pop(turn_step + 3)
                    elif challenged_move < 4:
                        win_challenge = self.resolve_challenge(
                            "CAPTAIN", actor, target, next_discard_step
                        )
                        if win_challenge:
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
                        win_challenge = False
                        if self.has_card("CAPTAIN", target):
                            win_challenge = self.resolve_challenge(
                                "CAPTAIN", actor, target, next_discard_step
                            )
                        else:
                            win_challenge = self.resolve_challenge(
                                "AMBASSADOR", actor, target, next_discard_step
                            )
                        if win_challenge:
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
                        win_challenge = self.resolve_challenge(
                            "CONTESSA", actor, target, next_discard_step
                        )
                        if win_challenge:
                            # reinstate blocked assassination
                            discard_turn = TURNS.index(f"discard-{victim}")
                            self.turn_list.insert(next_discard_step, discard_turn)
                            self.rewards[leader] += 5
                            self.rewards[victim] -= 5
                            self.reward_msgs.append(f"reward {actor} assassinate")
                            self.reward_msgs.append(f"punish {target} assassinated")
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
        truncations["__all__"] = env_truncation
        # seems that you cannot report one agent terminated with RLlib?
        # this is potentially the source of the single trajectory error
        # https://github.com/ray-project/ray/issues/10761
        dead = {agent: self.number_of_cards(agent) == 0 for agent in self.agents}
        players_left = [item[0] for item in dead.items() if item[1] == False]
        env_termination = len(players_left) == 1
        terminations = {agent: env_termination for agent in self.agents}
        terminations["__all__"] = env_termination

        if env_truncation or env_termination:
            self.rewards[players_left[0]] += 30
            self.game_ended = True

        for agent in self.agents:
            self.cumulative_rewards[agent] += self.rewards[agent]

        self.render(turn, players_left)

        return observations, self.rewards, terminations, truncations, infos

    def render(self, last_turn, players_left):
        """
        This function prints game data to the console after each step.
        """
        if not self.render_mode in ["moves", "games"]:
            return
        msg = ""
        if self.render_mode in ["moves"]:
            msg += f"----{TURNS[last_turn]}----\n"
            for agent in self.agents:
                msg += "{}: {}, coins {}, cards {}, rewards {} \n".format(
                    agent,
                    MOVES[self.state[agent]["MOVES"][-1]],
                    self.state[agent]["COINS"],
                    [card for card in self.state[agent]["CARDS"] if card],
                    self.rewards[agent],
                )
            if len(self.reward_msgs):
                msg += "--> " + str(self.reward_msgs) + "\n"
        if self.game_ended and self.render_mode in ["moves", "games"]:
            msg += f"\nGame Over - {players_left[0]} wins!\n"
            msg += f"Cumulative rewards: {self.cumulative_rewards}\n"
        print(msg)

    # game utils

    def has_card(self, card, player):
        """
        Returns true if player holds card, else false
        """
        return card in self.state[player]["CARDS"][:-2]

    def resolve_challenge(self, card, agent, target, discard_turn_step):
        """
        Reward players if they pass/fail challenge.
        Return True if challenge succeeds (target doesn't hold required card),
        or False if challenge fails (target does hold required card).
        """
        if self.has_card(card, target):
            self.rewards[target] += 5
            self.rewards[agent] -= 5
            self.reward_msgs.append(f"reward {target} win challenge")
            self.reward_msgs.append(f"punish {agent} lose challenge")
            discard_turn = TURNS.index(f"discard-{agent}")
            self.turn_list.insert(discard_turn_step, discard_turn)
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
            self.turn_list.insert(discard_turn_step, discard_turn)
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
        If no moves, return 'None' move (25).
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
        Return a player's last action, discounting 'None' move (25).
        If no moves, return 'None' move (25).
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
        Returns the number of cards a player has discounting empty slots ("").
        """
        cards = self.state[player]["CARDS"]
        count = 0
        for card in cards:
            if card:
                count += 1
        return count

    def get_counters(self):
        """
        Return a list of counter actions (e.g. block foreign aid) from
        the last step. Each counter is a dictionary with keys "agent" and "move".
        """
        counters = []
        for agent in self.agents:
            last_move = self.state[agent]["MOVES"][-1]
            if last_move in [16, 17, 18]:
                counters.append({"agent": agent, "move": last_move})
        return counters
