import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np
import random

device = torch.device('cpu')
loss_stat_low = []
from .utils import vaild_action_filter
from .utils import ACTIONS

class CNNEncoder(nn.Module):
    def __init__(self, ) -> None:
        super(CNNEncoder, self).__init__()

        self.cnn_layer1_3x3 = nn.Conv2d(19, 128, kernel_size=3, padding=1)
        self.batch_normal1 = nn.BatchNorm2d(128)
        self.cnn_layer2_3x3 = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
        self.batch_normal2 = nn.BatchNorm2d(128)
        self.cnn_layer3_3x3 = nn.Conv2d(128, 128, kernel_size=5, padding=4, dilation=2)
        self.batch_normal3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

    def forward(self, obs):
        """
        obs: torch.FloatTensor [B,C,H,W]
        features: [B,128]
        """

        low_level_features = self.cnn_layer1_3x3(obs)
        low_level_features = self.batch_normal1(low_level_features)
        low_level_features = self.relu(low_level_features)

        mid_level_features = self.cnn_layer2_3x3(low_level_features)
        mid_level_features = self.batch_normal2(mid_level_features)
        mid_level_features = self.relu(low_level_features + mid_level_features)

        high_level_features = self.cnn_layer3_3x3(mid_level_features)
        high_level_features = self.batch_normal3(high_level_features)
        high_level_features = self.relu(mid_level_features + high_level_features + low_level_features)

        self_id, self_x, self_y = torch.where(obs[:, -1].detach().data.cpu() == 1)
        features = high_level_features[self_id, :, self_x, self_y]
        return features


class low_QNet(nn.Module):
    def __init__(self) -> None:
        super(low_QNet, self).__init__()

        self.encoder = CNNEncoder()
        self.critic_layer = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.last_layer = nn.Sequential(
            nn.Linear(132, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, obs, mission_prob):
        mission_prob = mission_prob.view(1, -1)

        features = self.encoder(obs)
        temp = self.critic_layer(features)
        if temp.shape[0] != 1:
            org = copy.deepcopy(mission_prob)
            for size in range(temp.shape[0]-1):
                mission_prob = torch.cat((mission_prob, org), 0)

        temp = torch.cat((temp, mission_prob), 1)
        temp = F.softmax(temp)
        action = self.last_layer(temp)
        return action


class low_DQN(object):
    def __init__(self) -> None:

        self.gamma = 0.98
        self.eps = 0.2
        self.lr = 1e-3
        self.target_update = 10
        self.count = 1
        self.training = True
        self.mission = None

        self.q_net = low_QNet().to(device)
        self.target_q_net = low_QNet().to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

    def train(self, obs, action, next_obs, reward, done, mission):
        """
        obs: np.array shape = [B,C,H,W]
        action: np.array shape = [B]
        reward: np.array shape = [B]
        next_obs: np.array shape = [B,C,H,W]
        done: np.array shape = [B]
        """
        self.q_net.train()
        obs, action, reward, next_obs, done = self._prepare_input(obs, action, reward, next_obs, done)

        action = action.unsqueeze(1)
        q_values = self.q_net(obs, mission).gather(1, action).squeeze(1)


        max_action = self.q_net(next_obs, mission).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_q_net(next_obs, mission).gather(1, max_action)
        q_targets = reward + self.gamma * max_next_q_values * (1 - done)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        loss_stat_low.append(dqn_loss.item())
        self.optimizer.zero_grad()
        dqn_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 3)
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    def _prepare_input(self, obs, action, reward, next_obs, done):
        obs = torch.FloatTensor(obs).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        done = torch.FloatTensor(done).to(device)
        return obs, action, reward, next_obs, done

    def _prepare_action_input(self, obs):
        obs = torch.FloatTensor(obs).to(device)
        return obs

    @torch.no_grad()
    def action(self, obs, mission_prob, game_state):
        """
        :obs [C,H,W]
        :game_state The dictionary that describes everything on the board.
        这里需要检查是否合法
        """
        self.q_net.eval()
        self.mission = mission_prob

        valid_actions = vaild_action_filter(game_state)
        if self.training and np.random.random() < self.eps:
            obs = self._prepare_action_input(obs).unsqueeze(0)
            probs = self.q_net(obs, self.mission).squeeze(0)  # [action_space_size]

            invalid_actions = list(set([_ for _ in range(6)]) - set(valid_actions))
            for invalid_action in invalid_actions:
                probs[invalid_action] = -1e6
            probs = torch.softmax(probs, dim=0)

            action = np.random.choice(list(range(6)), p=probs.cpu().numpy())
        else:
            obs = self._prepare_action_input(obs).unsqueeze(0)
            probs = self.q_net(obs, self.mission).squeeze(0)  # [action_space_size]
            invalid_actions = list(set([_ for _ in range(6)]) - set(valid_actions))
            for invalid_action in invalid_actions:
                probs[invalid_action] = -1e8
            probs = torch.softmax(probs, dim=0)

            action = probs.argmax()
        return ACTIONS[action]

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        self.q_net.load_state_dict(state_dict)

    def save_model(self, model_path):
        state_dict = self.q_net.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].data.cpu()
        torch.save(state_dict, model_path)

    def switch_train_mode(self):
        self.q_net.train()
        self.training = True

    def switch_eval_mode(self):
        self.q_net.eval()
        self.training = False
