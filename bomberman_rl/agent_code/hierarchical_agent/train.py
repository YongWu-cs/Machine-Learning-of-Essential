from collections import namedtuple, deque
import numpy as np
from typing import List
import random

import events as e
from .callbacks import state_to_features_CNN
from .utils import ACTIONS, add_high_events, add_low_events

MEMORY_SIZE = 1024  # keep only ... last transitions
CUR_ROUND = 0
CUR_COIN = 0

all_old_bomb_state = None

high_reward = []
low_reward = []
score = []

# Events
# PLACEHOLDER_EVENT = "PLACEHOLDER"

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append([*args])

    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.high_memory = ReplayMemory(MEMORY_SIZE)
    self.low_memory = ReplayMemory(MEMORY_SIZE)
    self.batch_size = 64



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    global CUR_COIN, all_old_bomb_state
    if e.COIN_COLLECTED in events:
        CUR_COIN += 1
    if e.KILLED_OPPONENT in events:
        CUR_COIN += 5

    events = add_low_events(self, old_game_state, new_game_state, events, self_action, self.temp_mission, all_old_bomb_state)

    all_old_bomb_state = old_game_state['bombs']

    high_reward.append(reward_from_high_events(self, events, old_game_state['step']))

    self.high_memory.push(state_to_features_CNN(old_game_state),
                     self.temp_mission,
                     state_to_features_CNN(new_game_state),
                     reward_from_high_events(self, events, old_game_state['step']),
                     0)
    low_reward.append(reward_from_low_events(self, events, old_game_state['step']))
    self.low_memory.push(state_to_features_CNN(old_game_state),
                     ACTIONS.index(self_action),
                     state_to_features_CNN(new_game_state),
                     reward_from_low_events(self, events, old_game_state['step']),
                     1)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    global CUR_ROUND, CUR_COIN
    if e.COIN_COLLECTED in events:
        CUR_COIN += 1
    if e.KILLED_OPPONENT in events:
        CUR_COIN += 5

    events = add_low_events(self, last_game_state, last_game_state, events, last_action, self.temp_mission, all_old_bomb_state)

    high_reward.append(reward_from_high_events(self, events, last_game_state['step']))
    self.high_memory.push(state_to_features_CNN(last_game_state),
                          self.temp_mission,
                          state_to_features_CNN(last_game_state),
                          reward_from_high_events(self, events, last_game_state['step']),
                          0)
    low_reward.append(reward_from_low_events(self, events, last_game_state['step']))
    self.low_memory.push(state_to_features_CNN(last_game_state),
                         ACTIONS.index(last_action),
                         state_to_features_CNN(last_game_state),
                         reward_from_low_events(self, events, last_game_state['step']),
                         1)

    self.high_level_model.count = len(self.high_memory)

    for i in range(len(self.high_memory) - 2, -1, -1):
        if self.high_memory.memory[i][-1] == 1:
            break


    steps = 2 * (len(self.high_memory) // self.batch_size + 1)
    for _ in range(steps):
        memorys = self.high_memory.sample(self.batch_size)
        obs, mission, next_obs, reward, done = [], [], [], [], []
        for transition in memorys:
            obs.append(transition[0])
            mission.append(transition[1])
            next_obs.append(transition[2])
            reward.append(transition[3])
            done.append(transition[4])

        obs = np.stack(obs, axis=0)
        mission = np.array(mission)
        next_obs = np.stack(next_obs, axis=0)
        reward = np.array(reward)
        done = np.array(done)

    self.high_level_model.train(obs, mission, next_obs, reward, done)

    #########################################################################################################
    self.low_level_model.count = len(self.low_memory)
    for i in range(len(self.low_memory) - 2, -1, -1):
        if self.low_memory.memory[i][-1] == 1:
            break
    eps_i = i + 1
    while eps_i < len(self.low_memory):
        next_10_reward = 0
        discount = 0.9
        # 未来5步奖励
        for i in range(min(eps_i + 10, len(self.low_memory) - 1), eps_i - 1, -1):
            next_10_reward *= discount
            next_10_reward += self.low_memory.memory[i][3]
        self.low_memory.memory[eps_i][3] = next_10_reward
        eps_i += 1

    steps = 2 * (len(self.low_memory) // self.batch_size + 1)
    for _ in range(steps):
        memorys = self.low_memory.sample(self.batch_size)
        obs, action, next_obs, reward, done = [], [], [], [], []
        for transition in memorys:
            obs.append(transition[0])
            action.append(transition[1])
            next_obs.append(transition[2])
            reward.append(transition[3])
            done.append(transition[4])

        obs = np.stack(obs, axis=0)
        action = np.array(action)
        next_obs = np.stack(next_obs, axis=0)
        reward = np.array(reward)
        done = np.array(done)

    CUR_ROUND += 1
    self.logger.debug(f'{CUR_ROUND} done, coins is {CUR_COIN}.')
    CUR_COIN = 0

    self.low_level_model.train(obs, action, next_obs, reward, done, self.mission_prob)


    self.high_level_model.save_model("my-saved-high-model.pt")
    self.low_level_model.save_model("my-saved-low-model.pt")

    if high_reward:
        f = open("reward_high.txt", 'a')
        f.write(str(sum(high_reward) / len(high_reward))+" ")

        f2 = open("reward_low.txt", 'a')
        f2.write(str(sum(low_reward) / len(low_reward))+" ")

        f3 = open("score.txt", 'a')
        f3.write(str(CUR_COIN)+" ")

    high_reward.clear()
    low_reward.clear()




def reward_from_high_events(self, events: List[str], step) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.CRATE_DESTROYED: 4,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    if e.GOT_KILLED not in events:
        reward_sum += 1
    return reward_sum



def reward_from_low_events(self, events: List[str], step) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    if self.temp_mission == 0:
        game_rewards = {
            e.CLOSER_TO_COIN: 2,
            e.FURTHER_FROM_COIN: -1,
            e.GOT_KILLED: -5,
        }
        reward_sum = 0
        for event in events:
            if event in game_rewards:
                reward_sum += game_rewards[event]
        return reward_sum
    elif self.temp_mission == 1:
        game_rewards = {
            e.BOMB_DROPPED_GOOD: 2.5,
            e.BOMB_DROPPED_BAD: -2,
            e.GOT_KILLED: -5,
        }
        reward_sum = 0
        for event in events:
            if event in game_rewards:
                reward_sum += game_rewards[event]
        return reward_sum
    elif self.temp_mission == 3:
        game_rewards = {
            e.BOMB_DROPPED_GOOD: 2.5,
            e.BOMB_DROPPED_BAD: -2,
            e.GOT_KILLED: -5,
        }
        reward_sum = 0
        for event in events:
            if event in game_rewards:
                reward_sum += game_rewards[event]
        return reward_sum
    else:
        game_rewards = {
            e.ESCAPE: 2.5,
            e.SUICIDE: -2,
            e.GOT_KILLED: -5,
        }
        reward_sum = 0
        for event in events:
            if event in game_rewards:
                reward_sum += game_rewards[event]
        return reward_sum