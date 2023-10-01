import os

import numpy as np
import copy
import settings
from agent_code.hier_agent.high_level_model import high_DQN
from agent_code.hier_agent.low_level_model import low_DQN
import torch.nn.functional as F

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.high_level_model = high_DQN()
    # Release strategic aspects of missions, such as eating gold to kill opponents, depending on whether they complete strategic missions
    self.low_level_model = low_DQN()
    # The tactical level outputs action to calculate the current "distance" from the completion of the mission, such as eating gold if the gold is closer to the reward
    if self.train:
        if os.path.exists("my-saved-high-model.pt"):
            self.high_level_model.load_model("my-saved-high-model.pt")
            self.low_level_model.load_model("my-saved-low-model.pt")
    else:
        self.high_level_model.load_model("my-saved-high-model.pt")
        self.low_level_model.load_model("my-saved-low-model.pt")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    cnn_features = state_to_features_CNN(game_state)
    self.mission_prob = F.softmax(self.high_level_model.action(cnn_features))
    # print(self.mission_prob)
    self.temp_mission = copy.deepcopy(self.mission_prob).argmax()
    # print(self.temp_mission)
    # print()
    action = self.low_level_model.action(cnn_features, self.mission_prob, game_state)

    return action


last_game_state = None


def state_to_features_CNN(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    global last_game_state
    game_state_org = copy.deepcopy(game_state)
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None


    channels = []
    H, W = game_state['field'].shape
    name, score, bombs_left, (self_x, self_y) = game_state['self']
    # 1. wall
    wall = np.zeros((H, W))
    x, y = np.where(game_state['field'] == -1)
    for _x, _y in zip(x, y):
        wall[_x, _y] = 1
    channels.append(wall)
    # 2. crate
    crate = np.zeros((H, W))
    x, y = np.where(game_state['field'] == 1)
    for _x, _y in zip(x, y):
        crate[_x, _y] = 1
    channels.append(crate)
    # 3. bomb
    if last_game_state and last_game_state['bombs']:
        for bomb in last_game_state['bombs']:
            if bomb[-1] == 0:
                bomb = list(bomb)
                bomb[-1] = 1
                bomb = tuple(bomb)
                game_state['bombs'].append(bomb)

    area = np.zeros((H, W))
    bomb_direction = [copy.deepcopy(area) for _ in range(4)]
    bomb_channels = [copy.deepcopy(area) for _ in range(5)]
    min_bomb = None
    min_distance = 4
    for bomb in game_state['bombs']:
        (x, y), timer = bomb
        bomb_channels[timer][x, y] = 1
        distance = abs(self_x - x) + abs(self_y - y)
        if min_distance > distance:
            min_distance = distance
            min_bomb = [x, y]
    if min_bomb is not None and min_bomb[0] > self_x:
        bomb_direction[0][:] = 1
    if min_bomb is not None and min_bomb[0] < self_x:
        bomb_direction[1][:] = 1
    if min_bomb is not None and min_bomb[1] > self_y:
        bomb_direction[2][:] = 1
    if min_bomb is not None and min_bomb[1] < self_y:
        bomb_direction[3][:] = 1
    channels += bomb_channels
    channels += bomb_direction
    # 4. others
    others_channels = [copy.deepcopy(area) for _ in range(2)]
    for other in game_state['others']:
        name, score, bombs_left, (x, y) = other
        others_channels[0][x, y] = 1
        others_channels[1][x, y] = score
    channels += others_channels
    # 5. coin pos
    coin_area = np.zeros((H, W))
    coin_direction = [copy.deepcopy(coin_area) for _ in range(4)]
    min_distance = 1000
    min_coin = None
    for coin in game_state['coins']:
        coin_area[coin[0], coin[1]] = 1
        distance = abs(self_x - coin[0]) + abs(self_y - coin[1])
        if min_distance > distance:
            min_distance = distance
            min_coin = coin

    if min_coin is not None and min_coin[0] > self_x:
        coin_direction[0][:] = 1
    if min_coin is not None and min_coin[0] < self_x:
        coin_direction[1][:] = 1
    if min_coin is not None and min_coin[1] > self_y:
        coin_direction[2][:] = 1
    if min_coin is not None and min_coin[1] < self_y:
        coin_direction[3][:] = 1
    channels.append(coin_area)
    channels += coin_direction

    # 6. my pos
    my_area = np.zeros((H, W))
    name, score, bombs_left, (x, y) = game_state['self']
    my_area[x, y] = 1
    channels.append(my_area)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels, axis=0)


    last_game_state = game_state_org
    return stacked_channels
