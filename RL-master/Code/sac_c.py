"""
SAC (Soft Actor-Critic) 算法 - 连续动作版本

连续动作SAC的特点：
1. 处理连续动作空间，适用于机器人控制等任务
2. 使用重参数化技巧实现可微的随机策略
3. 通过tanh变换将动作限制在有界范围内
4. 需要修正tanh变换对概率密度的影响

核心技术：
- 重参数化采样：使随机策略可微分
- tanh变换：将无界高斯分布映射到有界动作空间
- 概率密度修正：补偿tanh变换对概率的影响
"""

import random
import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import common.rl_utils as rl_utils


class PolicyNetContinuous(torch.nn.Module):
    """
    连续动作的策略网络
    
    网络结构：
    - 共享特征层：提取状态特征
    - 均值头：输出动作分布的均值μ
    - 标准差头：输出动作分布的标准差σ
    
    输出：
    - 动作：通过重参数化采样得到
    - 对数概率：用于计算策略梯度和熵
    
    关键技术：
    1. 重参数化采样：ε ~ N(0,1), a = μ + σ * ε
    2. tanh变换：将无界动作映射到[-1,1]
    3. 动作缩放：乘以action_bound得到最终动作
    4. 概率密度修正：补偿tanh变换的雅可比行列式
    """
    
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        """
        初始化连续动作策略网络
        
        Args:
            state_dim (int): 状态空间维度
            hidden_dim (int): 隐藏层大小
            action_dim (int): 动作空间维度
            action_bound (float): 动作的最大绝对值
        """
        super(PolicyNetContinuous, self).__init__()
        
        # 共享特征提取层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        
        # 分别输出均值和标准差
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)    # 均值头
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)   # 标准差头
        
        # 动作边界，用于最终的动作缩放
        self.action_bound = action_bound

    def forward(self, x):
        """
        前向传播：状态 -> (动作, 对数概率)
        
        处理流程：
        1. 特征提取
        2. 计算高斯分布参数(μ, σ)
        3. 重参数化采样
        4. tanh变换和动作缩放
        5. 概率密度修正
        
        Args:
            x (torch.Tensor): 输入状态
            
        Returns:
            tuple: (action, log_prob)
                - action: 采样得到的动作
                - log_prob: 对应的对数概率密度
        """
        # 特征提取
        x = F.relu(self.fc1(x))
        
        # 计算高斯分布的参数
        mu = self.fc_mu(x)                    # 均值μ
        std = F.softplus(self.fc_std(x))      # 标准差σ，softplus确保σ > 0
        
        # 创建高斯分布
        dist = Normal(mu, std)
        
        # 重参数化采样：这是关键技术，使随机采样过程可微分
        # rsample() = μ + σ * ε，其中ε ~ N(0,1)
        # 相比sample()，rsample()保留了梯度信息
        normal_sample = dist.rsample()
        
        # 计算原始高斯分布的对数概率密度
        log_prob = dist.log_prob(normal_sample)
        
        # tanh变换：将无界的高斯样本映射到[-1, 1]
        # 这确保了动作在有界范围内
        action = torch.tanh(normal_sample)
        
        # 概率密度修正：补偿tanh变换对概率密度的影响
        # 根据变量变换公式：p_new(y) = p_old(x) / |dy/dx|
        # 对于tanh变换：d(tanh(x))/dx = 1 - tanh²(x)
        # 因此：log p_new = log p_old - log|1 - tanh²(x)|
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        
        # 动作缩放：从[-1, 1]缩放到[-action_bound, action_bound]
        action = action * self.action_bound
        
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    """
    连续动作的Q价值网络
    
    与离散动作版本的区别：
    - 输入：状态和动作的拼接，而非仅状态
    - 输出：单个Q值，而非所有动作的Q值向量
    - 结构：更深的网络以处理状态-动作的复杂关系
    
    网络设计原理：
    1. 状态-动作拼接：将s和a作为联合输入
    2. 多层结构：捕捉状态-动作间的非线性关系
    3. 单值输出：直接输出Q(s,a)的标量值
    
    在SAC中的作用：
    - 评估给定状态-动作对的价值
    - 为策略更新提供梯度信号
    - 通过双Q网络减少过估计偏差
    """
    
    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        初始化连续动作Q网络
        
        Args:
            state_dim (int): 状态空间维度
            hidden_dim (int): 隐藏层大小
            action_dim (int): 动作空间维度
        """
        super(QValueNetContinuous, self).__init__()
        
        # 第一层：处理状态-动作拼接
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        # 第二层：进一步特征提取
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # 输出层：单个Q值
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        """
        前向传播：(状态, 动作) -> Q值
        
        Args:
            x (torch.Tensor): 状态，形状为 [batch_size, state_dim]
            a (torch.Tensor): 动作，形状为 [batch_size, action_dim]
            
        Returns:
            torch.Tensor: Q值，形状为 [batch_size, 1]
        """
        # 将状态和动作在特征维度上拼接
        # 这样网络可以学习状态-动作的联合表示
        cat = torch.cat([x, a], dim=1)
        
        # 两层ReLU激活的全连接网络
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        
        # 输出单个Q值（无激活函数，因为Q值可以是任意实数）
        return self.fc_out(x)

class SACContinuous:
    """
    SAC连续动作版本
    
    与离散版本的主要区别：
    1. 策略网络：输出高斯分布参数而非概率向量
    2. Q网络：输入状态-动作对而非仅状态
    3. 熵计算：基于连续分布的微分熵
    4. 动作采样：使用重参数化技巧
    
    连续动作SAC的优势：
    - 自然处理连续控制问题
    - 不需要动作空间离散化
    - 保持动作的精确性和平滑性
    
    适用场景：
    - 机器人控制（关节角度、力矩）
    - 自动驾驶（转向角、加速度）
    - 游戏AI（连续移动、视角控制）
    """
    
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        """
        初始化连续动作SAC算法
        
        Args:
            state_dim (int): 状态空间维度
            hidden_dim (int): 神经网络隐藏层大小
            action_dim (int): 动作空间维度
            action_bound (float): 动作的最大绝对值
            actor_lr (float): Actor学习率
            critic_lr (float): Critic学习率
            alpha_lr (float): 温度参数学习率
            target_entropy (float): 目标熵，通常设为-action_dim
            tau (float): 软更新参数
            gamma (float): 折扣因子
            device: 计算设备
        """
        # ==================== 网络初始化 ====================
        # 策略网络：输出连续动作的高斯分布
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)
        
        # 双Q网络：处理连续状态-动作对
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        
        # 目标网络：提供稳定的学习目标
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        
        # 初始化目标网络参数
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        # ==================== 优化器初始化 ====================
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        # ==================== 温度参数α ====================
        # 连续动作的温度参数处理与离散版本相同
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        # ==================== 算法超参数 ====================
        self.target_entropy = target_entropy  # 连续动作的目标熵通常为-action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        """
        连续动作的动作选择
        
        与离散版本的区别：
        1. 返回连续值而非离散索引
        2. 需要处理多维动作空间
        3. 动作已经包含了探索噪声（来自高斯分布）
        
        Args:
            state: 环境状态
            
        Returns:
            list: 连续动作值列表（即使是1维也返回列表格式）
        """
        # 将状态转换为tensor格式
        state = torch.tensor(np.array([state], dtype=np.float32), dtype=torch.float).to(self.device)
        
        # 通过策略网络采样动作
        # actor返回(action, log_prob)，我们只需要action
        action = self.actor(state)[0]
        
        # 对于1维动作，返回标量值的列表
        # 对于多维动作，返回动作向量的列表
        if action.dim() == 2 and action.shape[1] == 1:
            return [action.item()]  # 1维动作
        else:
            return action.squeeze(0).cpu().numpy().tolist()  # 多维动作

    def calc_target(self, rewards, next_states, dones):
        """
        计算连续动作SAC的TD目标值
        
        连续动作版本的特点：
        1. 直接从策略网络采样下一状态的动作
        2. 熵计算基于连续分布的微分熵
        3. 不需要对所有动作求期望（因为是连续的）
        
        计算流程：
        1. 从当前策略采样下一状态的动作
        2. 计算对应的对数概率（用于熵计算）
        3. 使用双Q网络计算Q值
        4. 组合成SAC的价值函数：V = min(Q1, Q2) + α * H
        
        Args:
            rewards (torch.Tensor): 奖励批次
            next_states (torch.Tensor): 下一状态批次
            dones (torch.Tensor): 终止标志批次
            
        Returns:
            torch.Tensor: TD目标值
        """
        # 从当前策略采样下一状态的动作
        # 这里使用当前策略而非目标策略，这是SAC的特点
        next_actions, log_prob = self.actor(next_states)
        
        # 计算策略熵：对于连续分布，熵 = -log_prob
        # 注意：这里log_prob可能是多维的，需要求和
        entropy = -log_prob.sum(dim=1, keepdim=True)
        
        # 使用目标Q网络计算Q值
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        
        # SAC的价值函数：V(s) = min(Q1(s,a), Q2(s,a)) + α * H(π)
        # 使用双Q网络的最小值减少过估计
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        
        # TD目标：r + γ * V(s') * (1 - done)
        td_target = rewards + self.gamma * next_value * (1 - dones)
        
        return td_target

    def soft_update(self, net, target_net):
        """
        目标网络的软更新（与离散版本相同）
        
        软更新公式：θ_target = (1-τ) * θ_target + τ * θ_main
        
        Args:
            net: 主网络
            target_net: 目标网络
        """
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + 
                                    param.data * self.tau)

    def update(self, transition_dict):
        """
        连续动作SAC的更新函数
        
        更新流程与离散版本类似，但有以下区别：
        1. 动作是连续值而非离散索引
        2. 熵计算基于连续分布
        3. Q网络输入状态-动作对
        4. 包含奖励重塑（针对特定环境）
        
        Args:
            transition_dict (dict): 包含批次经验数据的字典
        """
        # ==================== 数据预处理 ====================
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        # ==================== 奖励重塑 ====================
        # 针对Pendulum环境的奖励重塑，将[-16, 0]映射到[0, 1]
        # 这有助于训练稳定性和收敛速度
        rewards = (rewards + 8.0) / 8.0

        # ==================== 更新Q网络 ====================
        """
        Q网络更新目标：最小化TD误差
        损失函数：L = E[(Q(s,a) - (r + γ * V(s')))²]
        """
        # 计算TD目标（包含熵正则化）
        td_target = self.calc_target(rewards, next_states, dones)
        
        # 第一个Q网络的更新
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        # 第二个Q网络的更新
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ==================== 更新策略网络 ====================
        """
        策略网络更新目标：最大化 E[Q(s,a)] + α * H(π)
        对于连续动作，直接从当前策略采样动作
        """
        # 从当前策略采样新动作
        new_actions, log_prob = self.actor(states)
        
        # 计算连续分布的熵：H = -log_prob
        # 对多维动作需要求和
        entropy = -log_prob.sum(dim=1, keepdim=True)
        
        # 计算新动作的Q值
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        
        # SAC的策略损失：最大化 E[Q(s,a)] + α * H(π)
        # 等价于最小化 -E[Q(s,a)] - α * H(π)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - 
                                torch.min(q1_value, q2_value))
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ==================== 更新温度参数α ====================
        """
        温度参数自动调节：使策略熵接近目标熵
        """
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # ==================== 软更新目标网络 ====================
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

# ==================== 连续动作SAC超参数配置 ====================
"""
连续动作SAC超参数说明：

环境特定参数：
- Pendulum-v1: 经典的倒立摆控制问题
- 状态空间: 3维连续 (cos(θ), sin(θ), θ_dot)
- 动作空间: 1维连续 [-2.0, 2.0] (扭矩)
- 奖励范围: [-16.27, 0] (越接近0越好)

学习率设置：
- actor_lr: 策略网络学习率，通常较小
- critic_lr: Q网络学习率，可以相对大一些
- alpha_lr: 温度参数学习率

算法参数：
- target_entropy: 连续动作的目标熵通常设为-action_dim
- tau: 软更新参数，控制目标网络更新速度
- gamma: 折扣因子，对连续控制任务通常设为0.99
"""

# 环境设置
env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode="rgb_array")
env.reset()
env.render()

# 获取环境参数
state_dim = env.observation_space.shape[0]  # 状态维度：3
action_dim = env.action_space.shape[0]      # 动作维度：1
action_bound = env.action_space.high[0]     # 动作边界：2.0

print(f"环境信息:")
print(f"  状态维度: {state_dim}")
print(f"  动作维度: {action_dim}")
print(f"  动作边界: [-{action_bound}, {action_bound}]")

# 设置随机种子
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# 算法超参数
actor_lr = 3e-4        # 策略网络学习率
critic_lr = 3e-3       # Q网络学习率
alpha_lr = 3e-4        # 温度参数学习率
num_episodes = 100     # 训练episode数量
hidden_dim = 128       # 神经网络隐藏层大小
gamma = 0.99           # 折扣因子，连续控制任务通常用0.99
tau = 0.005            # 软更新参数
buffer_size = 100000   # 经验回放缓冲区大小
minimal_size = 1000    # 开始训练的最小样本数
batch_size = 64        # 批次大小

# 连续动作的目标熵：通常设为-action_dim
target_entropy = -env.action_space.shape[0]  # -1

# 设备选择
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ==================== 智能体初始化和训练 ====================
print(f"\n开始连续动作SAC训练...")
print(f"目标熵: {target_entropy}")
print(f"设备: {device}")

# 初始化经验回放缓冲区和智能体
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device)

# 训练智能体
return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,
                                              replay_buffer, minimal_size,
                                              batch_size)

# ==================== 结果可视化和分析 ====================
"""
Pendulum环境的性能评估：
- 理论最优: 接近0的奖励
- 随机策略: 约-1600的奖励
- 良好性能: -200以上的奖励
"""

episodes_list = list(range(len(return_list)))

# 创建图形
plt.figure(figsize=(15, 5))

# 原始训练曲线
plt.subplot(1, 3, 1)
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {} (Raw)'.format(env_name))
plt.grid(True)

# 平滑训练曲线
plt.subplot(1, 3, 2)
mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {} (Smoothed)'.format(env_name))
plt.grid(True)

# 最后50个episode的分布
plt.subplot(1, 3, 3)
plt.hist(return_list[-50:], bins=10, alpha=0.7)
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.title('Last 50 Episodes Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()

# 保存结果
plt.savefig('sac_continuous_results.png', dpi=300, bbox_inches='tight')

# ==================== 性能统计 ====================
print(f"\n训练完成！")
print(f"最终50个episode平均奖励: {np.mean(return_list[-50:]):.2f}")
print(f"最高单episode奖励: {max(return_list):.2f}")
print(f"最低单episode奖励: {min(return_list):.2f}")
print(f"当前温度参数α: {agent.log_alpha.exp().item():.4f}")

# 性能评估
final_performance = np.mean(return_list[-50:])
if final_performance > -200:
    print("✅ 训练成功！智能体学会了良好的控制策略")
elif final_performance > -500:
    print("⚠️  训练部分成功，但仍有改进空间")
else:
    print("❌ 训练效果不佳，可能需要调整超参数")

print(f"\n环境解决标准参考:")
print(f"  优秀: > -200")
print(f"  良好: -200 ~ -500") 
print(f"  一般: -500 ~ -1000")
print(f"  较差: < -1000")