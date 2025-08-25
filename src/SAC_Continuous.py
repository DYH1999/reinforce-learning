import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils

# ============ 连续动作空间的SAC实现 ============

class PolicyNetContinuous(torch.nn.Module):
    """连续动作空间的策略网络
    SAC中的策略网络输出动作分布的参数，并使用重参数化技巧
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 输出动作均值
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        # 输出动作标准差的对数值
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # 动作边界

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))  # 确保标准差为正
        
        # 创建正态分布
        dist = Normal(mu, std)
        # 重参数化采样（rsample支持梯度回传）
        normal_sample = dist.rsample()
        # 计算对数概率
        log_prob = dist.log_prob(normal_sample)
        
        # 使用tanh将动作限制在[-1, 1]范围内
        action = torch.tanh(normal_sample)
        
        # 计算tanh变换后的对数概率密度
        # 这是SAC中的重要技巧（变量变换的雅可比修正），用于处理有界动作空间
        # 计算 tanh²(action)，1 - torch.tanh(action).pow(2): 雅可比行列式 |dy/dx|
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        
        # 缩放到实际动作范围
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    """连续动作空间的Q网络
    SAC使用两个Q网络来减少过估计偏差
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        # 输入是状态和动作的拼接
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # 将状态和动作拼接作为输入
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SACContinuous:
    """
    连续动作 SAC（带自动温度 & 双Q & 软更新）
    假设 self.actor(states) -> (actions, log_prob)
         Q(s,a) -> 标量 Q 值（按 batch 返回 [B,1]）
    """
    def __init__(
        self, state_dim, hidden_dim, action_dim, action_bound,
        actor_lr, critic_lr, alpha_lr, target_entropy,
        tau, gamma, device,
        reward_reshape=None  # 传入类似 lambda r: (r+8)/8 或 None
    ):
        # ---- 网络 ----
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # ---- 优化器 ----
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # ---- 自动温度 α（以 log_alpha 形式优化，数值稳定）----
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, device=device, requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = float(target_entropy)

        # ---- 超参 ----
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.reward_reshape = reward_reshape

    # ========== 交互动作（训练时：采样；评估时建议用均值动作） ==========
    def take_action(self, state):
        """
        训练用：随机采样动作（探索）
        评估时：建议用 actor.mean_action(state) 走确定性（如果你在 Actor 里提供该接口）
        """
        state = torch.as_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _logp = self.actor(state)     # 假设 actor 已做 tanh 限幅+log_prob 校正
        return action.squeeze(0).cpu().numpy().tolist()

    # ========== 计算 TD 目标 ==========
    @torch.no_grad()
    def calc_target(self, rewards, next_states, not_dones):
        """
        注意：这里用的是 not_dones（仅当 terminated=True 时为 0）
        truncated（时间截断）不应清零 bootstrap。
        """
        next_actions, next_logp = self.actor(next_states)  # 采样 a' 与 log π(a'|s')
        q1_t = self.target_critic_1(next_states, next_actions)
        q2_t = self.target_critic_2(next_states, next_actions)
        min_q_t = torch.min(q1_t, q2_t)
        alpha = self.log_alpha.exp()
        v_next = min_q_t - alpha * next_logp  # Q - α logπ  = V_soft
        td_target = rewards + self.gamma * not_dones * v_next
        return td_target

    # ========== 软更新 ==========
    def soft_update(self, net, target_net):
        for p_t, p in zip(target_net.parameters(), net.parameters()):
            p_t.data.copy_(p_t.data * (1.0 - self.tau) + p.data * self.tau)

    # ========== 核心更新 ==========
    def update(self, transition_dict):
        # ------ 准备 batch ------
        states = torch.as_tensor(transition_dict['states'],  dtype=torch.float, device=self.device)
        actions = torch.as_tensor(transition_dict['actions'], dtype=torch.float, device=self.device)
        rewards = torch.as_tensor(transition_dict['rewards'], dtype=torch.float, device=self.device).unsqueeze(-1)
        next_states = torch.as_tensor(transition_dict['next_states'], dtype=torch.float, device=self.device)
        # 关键：用 not_dones（= 1 - terminated），不要把 truncated 当终止
        # 如果你现在存的是 dones = terminated or truncated，请改采样器存 not_dones：
        #   not_dones = 1.0 - terminated
        # 这里先兼容：如果传进来的是 dones（0/1），就当作 not_dones = 1 - dones
        dones_like = torch.as_tensor(transition_dict['dones'], dtype=torch.float, device=self.device).unsqueeze(-1)
        not_dones = 1.0 - dones_like  # 强烈建议你在收集时区分 terminated / truncated

        # 可选：奖励重塑
        if self.reward_reshape is not None:
            rewards = self.reward_reshape(rewards)

        # ------ 更新两个 Critic ------
        with torch.no_grad():
            td_target = self.calc_target(rewards, next_states, not_dones)

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        critic_1_loss = F.mse_loss(q1, td_target)
        critic_2_loss = F.mse_loss(q2, td_target)

        self.critic_1_optimizer.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ------ 更新 Actor ------
        new_actions, log_prob = self.actor(states)   # 采样 a ~ π(a|s)
        q1_pi = self.critic_1(states, new_actions)
        q2_pi = self.critic_2(states, new_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        alpha_detached = self.log_alpha.exp().detach()  # 不让梯度回传到 α
        actor_loss = (alpha_detached * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------ 自动温度 α ------
        # 正确的 α 损失（优化 log_alpha）：使 E[ -logπ ] ≈ target_entropy
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # ------ 软更新目标网络 ------
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


