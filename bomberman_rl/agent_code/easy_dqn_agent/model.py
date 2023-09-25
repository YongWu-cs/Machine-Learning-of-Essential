import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from .utils import ACTIONS

class ConvolutionalQnet(torch.nn.Module):
    def __init__(self, action_dim, in_channels=6):
        #模型就是进行三次卷积之后两次线性层
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(11 * 11 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x=x.flatten(1)
        x = F.relu(self.fc4(x))
        return self.head(x)

# DQN 算法实现
class DQN(object):
    def __init__(self) -> None:
        self.state_dim=6
        self.action_dim=6
        self.gamma = 0.98  # 折扣因子
        self.eps = 0.2  # eps贪心策略
        self.lr = 1e-3
        self.target_update = 10  # 目标网络更新频率
        self.count = 1
        self.training = True

        self.q_net = ConvolutionalQnet(self.action_dim,in_channels=self.state_dim).to(device)
        self.target_q_net = ConvolutionalQnet(self.action_dim,in_channels=self.state_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

    def train(self, obs, action, next_obs, reward, done):
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
        q_values = self.q_net(obs).gather(1, action).squeeze()  # [B]
        # 下个状态最大Q值
        # max_next_q_values = self.target_q_net(next_obs).max(1)[0].view(-1) # [B]

        max_action = self.q_net(next_obs).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_q_net(next_obs).gather(1, max_action).squeeze()
        q_targets = reward + self.gamma * max_next_q_values * (1 - done)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        print("loss is:{}".format(dqn_loss))
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
    def action(self, obs, game_state):
        """
        :obs [C,H,W]
        :game_state The dictionary that describes everything on the board.
        """
        self.q_net.eval()
        if self.training and np.random.random() < self.eps:
            obs = self._prepare_action_input(obs).unsqueeze(0)
            probs = self.q_net(obs).squeeze()  # [action_space_size]
            probs = torch.softmax(probs, dim=0)
            action = np.random.choice(list(range(6)), p=probs.cpu().numpy())
        else:
            obs = self._prepare_action_input(obs).unsqueeze(0)
            probs = self.q_net(obs)  # [action_space_size]
            probs = torch.softmax(probs, dim=1).squeeze()
            action = np.random.choice(list(range(6)), p=probs.cpu().numpy())
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
