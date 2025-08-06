"""
SAC (Soft Actor-Critic) 算法 - 离散动作版本

SAC是基于最大熵强化学习的off-policy算法，主要特点：
1. 最大熵目标：在最大化回报的同时最大化策略熵，鼓励探索
2. 双Q网络：减少价值估计的过估计偏差
3. 自动温度调节：自适应调整探索与利用的平衡
4. 软更新：目标网络的参数软更新，提高训练稳定性

核心思想：
- 将熵正则化引入强化学习目标函数
- 学习随机策略而非确定性策略
- 通过温度参数α控制探索程度
"""

import random
import gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import common.rl_utils as rl_utils


class PolicyNet(torch.nn.Module):
    """
    SAC策略网络 - 输出离散动作的概率分布
    
    与PPO的策略网络类似，但在SAC中有不同的作用：
    1. 用于计算策略熵，实现最大熵目标
    2. 直接参与Q值的期望计算
    3. 支持连续的策略改进
    
    网络结构：简单的两层全连接网络
    """
    
    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        初始化策略网络
        
        Args:
            state_dim (int): 状态空间维度
            hidden_dim (int): 隐藏层神经元数量
            action_dim (int): 动作空间维度（离散动作数量）
        """
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        前向传播：状态 -> 动作概率分布
        
        Args:
            x (torch.Tensor): 输入状态
            
        Returns:
            torch.Tensor: 动作概率分布，用于计算熵和期望Q值
        """
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class QValueNet(torch.nn.Module):
    """
    Q价值网络 - 估计状态-动作价值函数Q(s,a)
    
    在SAC中的特殊作用：
    1. 双Q网络设计：使用两个独立的Q网络减少过估计
    2. 目标网络：使用软更新的目标网络提高训练稳定性
    3. 熵正则化：Q值计算中包含策略熵项
    
    网络结构：简单的两层全连接网络
    输出：对于每个状态，输出所有动作的Q值
    """
    
    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        初始化Q价值网络
        
        Args:
            state_dim (int): 状态空间维度
            hidden_dim (int): 隐藏层神经元数量
            action_dim (int): 动作空间维度
        """
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        前向传播：状态 -> 所有动作的Q值
        
        Args:
            x (torch.Tensor): 输入状态，形状为 [batch_size, state_dim]
            
        Returns:
            torch.Tensor: Q值，形状为 [batch_size, action_dim]
                         每行表示在对应状态下所有动作的Q值
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SAC:
    """
    SAC (Soft Actor-Critic) 算法 - 离散动作版本
    
    SAC算法的核心组件：
    1. Actor网络：学习随机策略π(a|s)
    2. 双Critic网络：学习Q函数，减少过估计偏差
    3. 目标网络：提供稳定的学习目标
    4. 温度参数α：自动调节探索与利用的平衡
    
    算法特点：
    - 最大熵目标：J = E[R + αH(π)]，其中H(π)是策略熵
    - 双Q学习：使用两个Q网络的最小值作为目标
    - 软更新：目标网络参数的指数移动平均更新
    - 自动温度调节：通过梯度下降优化温度参数
    """
    
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        """
        初始化SAC算法
        
        Args:
            state_dim (int): 状态空间维度
            hidden_dim (int): 神经网络隐藏层大小
            action_dim (int): 动作空间维度
            actor_lr (float): Actor学习率
            critic_lr (float): Critic学习率
            alpha_lr (float): 温度参数学习率
            target_entropy (float): 目标熵值，通常设为-action_dim
            tau (float): 软更新参数，控制目标网络更新速度
            gamma (float): 折扣因子
            device: 计算设备
        """
        # ==================== 网络初始化 ====================
        # 策略网络（Actor）
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        
        # 双Q网络设计：使用两个独立的Q网络
        # 这是SAC的重要特性，用于减少Q值的过估计偏差
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        
        # 目标网络：用于计算稳定的TD目标
        # 通过软更新机制缓慢跟踪主网络
        self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        
        # 初始化目标网络参数与主网络相同
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        # ==================== 优化器初始化 ====================
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        # ==================== 温度参数α的自动调节 ====================
        # 使用log(α)而不是α本身的优势：
        # 1. 确保α始终为正数（exp(log_α) > 0）
        # 2. 数值稳定性更好
        # 3. 梯度更新更稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 允许对α进行梯度更新
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        # ==================== 算法超参数 ====================
        self.target_entropy = target_entropy  # 目标熵：期望的策略随机性水平
        self.gamma = gamma                     # 折扣因子
        self.tau = tau                         # 软更新参数
        self.device = device

    def take_action(self, state):
        """
        根据当前策略选择动作
        
        SAC中的动作选择特点：
        1. 直接从学习到的随机策略中采样
        2. 不需要额外的探索机制（如ε-greedy）
        3. 策略本身就包含了适当的随机性
        
        这与确定性策略算法（如DDPG）不同，SAC学习的是随机策略，
        天然具备探索能力。
        
        Args:
            state: 环境状态
            
        Returns:
            int: 选择的动作索引
        """
        # 将状态转换为tensor格式
        state = torch.tensor(np.array([state], dtype=np.float32), dtype=torch.float).to(self.device)
        
        # 通过策略网络获得动作概率分布
        probs = self.actor(state)
        
        # 从概率分布中采样动作
        # SAC的策略是随机的，直接采样即可获得探索
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        return action.item()

    def calc_target(self, rewards, next_states, dones):
        """
        计算SAC的TD目标值
        
        SAC的TD目标包含两个关键组件：
        1. Q值的期望：E[Q(s',a')] = Σ π(a'|s') * Q(s',a')
        2. 策略熵：H(π) = -Σ π(a'|s') * log π(a'|s')
        
        最终目标：r + γ * (E[Q(s',a')] + α * H(π)) * (1 - done)
        
        这体现了SAC的核心思想：在最大化回报的同时最大化策略熵
        
        Args:
            rewards (torch.Tensor): 奖励批次
            next_states (torch.Tensor): 下一状态批次
            dones (torch.Tensor): 终止标志批次
            
        Returns:
            torch.Tensor: TD目标值
        """
        # 获取下一状态的动作概率分布
        next_probs = self.actor(next_states)
        
        # 计算策略熵：H(π) = -Σ π(a|s) * log π(a|s)
        # 添加小常数1e-8防止log(0)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        
        # 获取两个目标Q网络的Q值
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        
        # 双Q学习：取两个Q值的最小值，减少过估计偏差
        # 计算Q值的期望：E[Q(s',a')] = Σ π(a'|s') * min(Q1(s',a'), Q2(s',a'))
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1, keepdim=True)
        
        # SAC的价值函数：V(s) = E[Q(s,a)] + α * H(π)
        # 这里α = exp(log_alpha)，确保α > 0
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        
        # TD目标：r + γ * V(s') * (1 - done)
        td_target = rewards + self.gamma * next_value * (1 - dones)
        
        return td_target

    def soft_update(self, net, target_net):
        """
        目标网络的软更新
        
        软更新公式：θ_target = (1-τ) * θ_target + τ * θ_main
        
        软更新的优势：
        1. 提供稳定的学习目标：目标网络变化缓慢，避免学习目标剧烈变化
        2. 减少训练不稳定性：相比硬更新（直接复制），软更新更平滑
        3. 提高样本效率：目标网络能够更好地利用历史信息
        
        参数τ的作用：
        - τ接近0：目标网络更新很慢，更稳定但可能过时
        - τ接近1：目标网络更新很快，接近硬更新
        - 通常设置τ=0.005左右
        
        Args:
            net: 主网络（被更新的网络）
            target_net: 目标网络（提供稳定目标的网络）
        """
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 指数移动平均更新：θ_target = (1-τ) * θ_target + τ * θ_main
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + 
                                    param.data * self.tau)

    def update(self, transition_dict):
        """
        SAC算法的核心更新函数
        
        更新顺序：
        1. 更新两个Q网络（Critic）
        2. 更新策略网络（Actor）
        3. 更新温度参数α
        4. 软更新目标网络
        
        Args:
            transition_dict (dict): 包含批次经验数据的字典
        """
        # ==================== 数据预处理 ====================
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # ==================== 更新Q网络 ====================
        """
        Q网络更新目标：最小化TD误差
        损失函数：L = E[(Q(s,a) - (r + γ * V(s')))²]
        其中V(s') = E[Q(s',a')] + α * H(π)
        """
        # 计算TD目标（包含熵正则化项）
        td_target = self.calc_target(rewards, next_states, dones)
        
        # 第一个Q网络的更新
        critic_1_q_values = self.critic_1(states).gather(1, actions)  # 选择执行动作的Q值
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
        
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        # 第二个Q网络的更新（独立更新，减少过估计）
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))
        
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ==================== 更新策略网络 ====================
        """
        策略网络更新目标：最大化 E[Q(s,a)] + α * H(π)
        损失函数：L = -E[Q(s,a)] - α * H(π)
        """
        # 获取当前策略的动作概率分布
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        
        # 计算策略熵：H(π) = -Σ π(a|s) * log π(a|s)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        
        # 获取当前Q值
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        
        # 计算Q值的期望：E[Q(s,a)] = Σ π(a|s) * min(Q1(s,a), Q2(s,a))
        # 使用双Q网络的最小值减少过估计
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1, keepdim=True)
        
        # SAC的策略损失：最大化 E[Q(s,a)] + α * H(π)
        # 等价于最小化 -E[Q(s,a)] - α * H(π)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ==================== 更新温度参数α ====================
        """
        温度参数自动调节：
        目标是使策略熵接近目标熵target_entropy
        损失函数：L_α = E[-α * (log π(a|s) + target_entropy)]
        """
        # α的损失函数：希望当前熵接近目标熵
        # 当熵高于目标时，增大α以减少熵；当熵低于目标时，减小α以增加熵
        alpha_loss = torch.mean((entropy - target_entropy).detach() * self.log_alpha.exp())
        
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # ==================== 软更新目标网络 ====================
        # 缓慢更新目标网络，提供稳定的学习目标
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


# ==================== SAC算法超参数配置 ====================
"""
SAC超参数说明：

学习率设置：
- actor_lr: 策略网络学习率
- critic_lr: Q网络学习率，通常比actor_lr大
- alpha_lr: 温度参数学习率，控制熵权重的调节速度

算法参数：
- gamma: 折扣因子，控制未来奖励的重要性
- tau: 软更新参数，控制目标网络更新速度
- target_entropy: 目标熵，通常设为-action_dim（离散动作）

经验回放参数：
- buffer_size: 回放缓冲区大小
- minimal_size: 开始训练的最小样本数
- batch_size: 每次更新的批次大小
"""

actor_lr = 1e-3        # 策略网络学习率
critic_lr = 1e-2       # Q网络学习率，相对较大
alpha_lr = 1e-2        # 温度参数学习率
num_episodes = 200     # 训练episode数量
hidden_dim = 128       # 神经网络隐藏层大小
gamma = 0.98           # 折扣因子
tau = 0.005            # 软更新参数，较小的值确保稳定更新
buffer_size = 10000    # 经验回放缓冲区大小
minimal_size = 500     # 开始训练的最小样本数
batch_size = 64        # 批次大小
target_entropy = -1    # 目标熵，对于2个动作通常设为-1

# 设备选择
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ==================== 环境设置和智能体初始化 ====================
"""
CartPole-v1环境配置：
- 状态空间：4维连续空间
- 动作空间：2维离散空间
- SAC的优势：不需要额外的探索策略，策略本身就是随机的
"""

env_name = 'CartPole-v1'
env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset()
env.render()

# 设置随机种子确保实验可重复
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# 初始化经验回放缓冲区
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

# 获取环境参数
state_dim = env.observation_space.shape[0]  # 状态维度：4
action_dim = env.action_space.n             # 动作维度：2

# 初始化SAC智能体
agent = SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr,
            target_entropy, tau, gamma, device)

# ==================== 训练过程 ====================
"""
SAC训练流程：
1. 智能体与环境交互，收集经验存入回放缓冲区
2. 从缓冲区采样批次数据
3. 更新Q网络、策略网络和温度参数
4. 软更新目标网络
"""

print("开始SAC训练...")
print(f"环境: {env_name}")
print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
print(f"目标熵: {target_entropy}")
print(f"设备: {device}")

# 使用off-policy训练函数训练智能体
return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,
                                              replay_buffer, minimal_size,
                                              batch_size)

# ==================== 结果可视化 ====================
"""
训练结果分析：
1. 原始奖励曲线：显示训练过程中的性能变化
2. 平滑奖励曲线：更清晰地观察学习趋势
"""

episodes_list = list(range(len(return_list)))

# 创建图形
plt.figure(figsize=(12, 4))

# 绘制原始训练曲线
plt.subplot(1, 2, 1)
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {} (Raw)'.format(env_name))
plt.grid(True)

# 绘制平滑后的训练曲线
plt.subplot(1, 2, 2)
mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {} (Smoothed)'.format(env_name))
plt.grid(True)

plt.tight_layout()
plt.show()

# 保存结果图像
plt.savefig('sac_results.png', dpi=300, bbox_inches='tight')

# 打印训练结果统计
print(f"\n训练完成！")
print(f"最终100个episode平均奖励: {np.mean(return_list[-100:]):.2f}")
print(f"最高单episode奖励: {max(return_list):.2f}")
print(f"当前温度参数α: {agent.log_alpha.exp().item():.4f}")
print(f"训练是否成功 (≥475): {'是' if np.mean(return_list[-100:]) >= 475 else '否'}")