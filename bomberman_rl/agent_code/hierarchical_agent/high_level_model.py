import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cpu')


loss_stat_high = []
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


class high_QNet(nn.Module):
    def __init__(self) -> None:
        super(high_QNet, self).__init__()

        self.encoder = CNNEncoder()
        self.critic_layer = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, obs):
        features = self.encoder(obs)
        logits = self.critic_layer(features)
        return logits



class high_DQN(object):
    def __init__(self) -> None:

        self.gamma = 0.98
        self.eps = 0.2
        self.lr = 1e-3
        self.target_update = 10
        self.count = 1
        self.training = True

        self.q_net = high_QNet().to(device)
        self.target_q_net = high_QNet().to(device)
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
        q_values = self.q_net(obs).gather(1, action).squeeze(1)


        max_action = self.q_net(next_obs).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_q_net(next_obs).gather(1, max_action)
        q_targets = reward + self.gamma * max_next_q_values * (1 - done)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        loss_stat_high.append(dqn_loss.item())

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
    def action(self, obs):
        """
        :obs [C,H,W]
        :game_state The dictionary that describes everything on the board.
        这里需要检查是否合法
        """
        self.q_net.eval()
        obs = self._prepare_action_input(obs).unsqueeze(0)
        probs = self.q_net(obs).squeeze(0)  # [action_space_size]
        return probs


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
