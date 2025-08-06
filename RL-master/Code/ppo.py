"""
PPO (Proximal Policy Optimization) 算法实现

PPO是OpenAI提出的策略梯度算法，是目前最流行的强化学习算法之一。
主要特点：
1. 简单易实现，相比TRPO避免了复杂的二阶优化
2. 样本效率高，通过重要性采样重复利用数据
3. 训练稳定，通过截断机制防止策略更新过大
4. 适用范围广，在连续和离散动作空间都表现良好

核心思想：
- 使用重要性采样允许off-policy更新
- 通过截断(clipping)限制策略更新幅度
- 结合Actor-Critic架构同时学习策略和价值函数
"""

import gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import common.rl_utils as rl_utils


class PolicyNet(torch.nn.Module):
    """
    策略网络 (Actor) - PPO算法的核心组件之一
    
    功能：将状态映射为动作概率分布
    架构：简单的两层全连接网络
    输出：softmax概率分布，确保所有动作概率和为1
    
    在Actor-Critic框架中的作用：
    - Actor：负责策略学习，输出动作选择的概率
    - 通过策略梯度方法优化，目标是最大化期望回报
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
        # 第一层：状态特征提取
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层：动作概率输出
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        前向传播：状态 -> 动作概率分布
        
        Args:
            x (torch.Tensor): 输入状态，形状为 [batch_size, state_dim]
            
        Returns:
            torch.Tensor: 动作概率分布，形状为 [batch_size, action_dim]
                         每行和为1，表示在对应状态下各动作的选择概率
        """
        # 第一层 + ReLU激活
        x = F.relu(self.fc1(x))
        # 第二层 + Softmax归一化得到概率分布
        # dim=1表示在动作维度上进行softmax
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    """
    价值网络 (Critic) - PPO算法的核心组件之二
    
    功能：估计状态价值函数V(s)，即在状态s下遵循当前策略的期望回报
    架构：简单的两层全连接网络
    输出：标量值，表示状态的价值估计
    
    在Actor-Critic框架中的作用：
    - Critic：负责价值评估，为Actor提供学习信号
    - 用于计算优势函数A(s,a) = Q(s,a) - V(s)
    - 通过最小化TD误差进行优化
    """
    
    def __init__(self, state_dim, hidden_dim):
        """
        初始化价值网络
        
        Args:
            state_dim (int): 状态空间维度
            hidden_dim (int): 隐藏层神经元数量
        """
        super(ValueNet, self).__init__()
        # 第一层：状态特征提取
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层：价值输出（单个标量）
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        前向传播：状态 -> 价值估计
        
        Args:
            x (torch.Tensor): 输入状态，形状为 [batch_size, state_dim]
            
        Returns:
            torch.Tensor: 状态价值估计，形状为 [batch_size, 1]
                         表示在对应状态下的期望累积回报
        """
        # 第一层 + ReLU激活
        x = F.relu(self.fc1(x))
        # 第二层直接输出价值（无激活函数，因为价值可以是任意实数）
        return self.fc2(x)


class PPO:
    """
    PPO (Proximal Policy Optimization) 算法实现
    
    PPO算法的核心创新：
    1. 重要性采样：允许使用旧策略收集的数据更新新策略
    2. 截断机制：防止策略更新过大，保持训练稳定性
    3. 多轮更新：对同一批数据进行多次更新，提高样本效率
    
    算法流程：
    1. 收集一批轨迹数据
    2. 计算优势函数
    3. 对数据进行多轮PPO更新
    4. 重复上述过程
    
    关键公式：
    - 重要性采样比率：r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    - PPO截断目标：L^CLIP = min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)
    """
    
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        """
        初始化PPO算法
        
        Args:
            state_dim (int): 状态空间维度
            hidden_dim (int): 神经网络隐藏层大小
            action_dim (int): 动作空间维度
            actor_lr (float): Actor学习率
            critic_lr (float): Critic学习率
            lmbda (float): GAE中的λ参数，控制偏差-方差权衡
            epochs (int): 每批数据的更新轮数
            eps (float): PPO截断参数ε，控制策略更新幅度
            gamma (float): 折扣因子
            device: 计算设备（CPU或GPU）
        """
        # 初始化Actor和Critic网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        
        # 保存网络参数
        self.state_dim = state_dim
        
        # 初始化优化器 - 使用不同学习率分别优化Actor和Critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # PPO算法超参数
        self.gamma = gamma      # 折扣因子，控制未来奖励的重要性
        self.lmbda = lmbda      # GAE参数，控制优势估计的偏差-方差权衡
        self.epochs = epochs    # 每批数据的更新轮数，提高样本利用效率
        self.eps = eps          # 截断参数，防止策略更新过大
        self.device = device

    def take_action(self, state):
        """
        根据当前策略选择动作
        
        动作选择过程：
        1. 将状态转换为tensor并送入策略网络
        2. 获得动作概率分布
        3. 从概率分布中采样动作
        
        这种随机采样的好处：
        - 保持探索能力，避免过早收敛到局部最优
        - 符合策略梯度算法的理论要求
        - 在训练过程中提供必要的随机性
        
        Args:
            state: 环境状态，numpy数组格式
            
        Returns:
            int: 选择的动作索引
        """
        # 将状态转换为tensor格式
        # [state] -> [[state]] 增加batch维度
        # view(1, self.state_dim) 确保形状正确
        state = torch.tensor([state], dtype=torch.float).to(self.device).view(1, self.state_dim)
        
        # 通过策略网络获得动作概率分布
        probs = self.actor(state)
        
        # 创建分类分布对象
        # Categorical分布适用于离散动作空间
        action_dist = torch.distributions.Categorical(probs)
        
        # 从概率分布中采样动作
        # sample()方法根据概率进行随机采样
        action = action_dist.sample()
        
        # 返回标量动作值
        return action.item()

    def update(self, transition_dict):
        """
        PPO算法的核心更新函数
        
        更新流程：
        1. 数据预处理：将轨迹数据转换为tensor
        2. 计算TD目标和优势函数
        3. 多轮PPO更新：重复利用数据提高样本效率
        
        PPO的关键创新：
        - 重要性采样：使用旧策略数据更新新策略
        - 截断机制：防止策略更新过大
        - 多轮更新：提高样本利用效率
        
        Args:
            transition_dict (dict): 包含轨迹数据的字典
                - states: 状态序列
                - actions: 动作序列
                - rewards: 奖励序列
                - next_states: 下一状态序列
                - dones: 终止标志序列
        """
        # ==================== 数据预处理 ====================
        # 将轨迹数据转换为tensor格式
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        # ==================== 计算TD目标和优势函数 ====================
        # TD目标：r + γV(s') * (1 - done)
        # (1 - dones)确保终止状态的价值为0
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        
        # TD误差：δ = r + γV(s') - V(s)
        # 这是优势函数的无偏估计
        td_delta = td_target - self.critic(states)
        
        # 计算GAE优势函数
        # 结合多步TD误差，在偏差和方差间取得平衡
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        
        # 计算旧策略的对数概率（用于重要性采样）
        # gather(1, actions)选择对应动作的概率
        # detach()阻止梯度传播，因为这是"旧"策略的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        
        # ==================== 多轮PPO更新 ====================
        # 对同一批数据进行多轮更新，提高样本利用效率
        for _ in range(self.epochs):
            # 计算当前策略的对数概率
            log_probs = torch.log(self.actor(states).gather(1, actions))
            
            # 重要性采样比率：π_new(a|s) / π_old(a|s)
            # exp(log_prob_new - log_prob_old) = prob_new / prob_old
            ratio = torch.exp(log_probs - old_log_probs)
            
            # PPO的两个代理目标
            # surr1: 标准策略梯度目标
            surr1 = ratio * advantage
            
            # surr2: 截断版本，限制比率在[1-ε, 1+ε]范围内
            # 防止策略更新过大，保持训练稳定性
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            
            # PPO损失函数：取两个目标的最小值
            # 这确保了保守的策略更新
            # 负号是因为我们要最大化目标，但优化器执行最小化
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            
            # Critic损失：TD目标与价值估计的均方误差
            # detach()防止梯度传播到TD目标的计算
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            # ==================== 梯度更新 ====================
            # 清零梯度
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            
            # 更新参数
            self.actor_optimizer.step()
            self.critic_optimizer.step()

# ==================== PPO算法超参数配置 ====================
"""
PPO超参数说明：

学习率设置：
- actor_lr: Actor网络学习率，通常比Critic小，因为策略更新需要更谨慎
- critic_lr: Critic网络学习率，可以相对大一些，因为价值函数学习相对稳定

算法参数：
- gamma: 折扣因子，控制未来奖励的重要性，接近1重视长期回报
- lmbda: GAE参数，控制优势估计的偏差-方差权衡
- epochs: 每批数据的更新轮数，提高样本利用效率
- eps: PPO截断参数，防止策略更新过大，保持训练稳定性

网络结构：
- hidden_dim: 隐藏层大小，影响网络的表达能力
"""

actor_lr = 1e-3      # Actor学习率：较小的学习率确保策略更新稳定
critic_lr = 1e-2     # Critic学习率：可以相对大一些
num_episodes = 500   # 训练episode数量
hidden_dim = 128     # 神经网络隐藏层大小
gamma = 0.98         # 折扣因子：重视长期回报
lmbda = 0.95         # GAE参数：在偏差和方差间取得平衡
epochs = 10          # 每批数据更新轮数：提高样本效率
eps = 0.2            # PPO截断参数：防止策略更新过大

# 设备选择：优先使用GPU加速训练
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ==================== 环境设置和智能体初始化 ====================
"""
CartPole-v1环境：
- 状态空间：4维连续空间（位置、速度、角度、角速度）
- 动作空间：2维离散空间（向左推、向右推）
- 目标：保持杆子平衡尽可能长的时间
- 最大步数：500步
- 成功标准：连续100个episode平均奖励≥475
"""

env_name = 'CartPole-v1'
env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset()
env.render()

# 设置随机种子确保实验可重复
torch.manual_seed(0)

# 获取环境的状态和动作空间维度
state_dim = env.observation_space.shape[0]  # 状态维度：4
action_dim = env.action_space.n             # 动作维度：2

# 初始化PPO智能体
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

# ==================== 训练过程 ====================
"""
训练流程：
1. 智能体与环境交互收集轨迹数据
2. 计算优势函数和TD目标
3. 使用PPO算法更新策略和价值网络
4. 重复上述过程直到收敛
"""

print("开始PPO训练...")
print(f"环境: {env_name}")
print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
print(f"设备: {device}")

# 使用on-policy训练函数训练智能体
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

# ==================== 结果可视化 ====================
"""
训练结果分析：
1. 原始奖励曲线：显示每个episode的累积奖励
2. 平滑奖励曲线：使用移动平均减少噪声，更清晰地观察学习趋势
"""

episodes_list = list(range(len(return_list)))

# 绘制原始训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {} (Raw)'.format(env_name))
plt.grid(True)

# 绘制平滑后的训练曲线
plt.subplot(1, 2, 2)
mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {} (Smoothed)'.format(env_name))
plt.grid(True)

plt.tight_layout()
plt.show()

# 打印训练结果统计
print(f"\n训练完成！")
print(f"最终100个episode平均奖励: {np.mean(return_list[-100:]):.2f}")
print(f"最高单episode奖励: {max(return_list):.2f}")
print(f"训练是否成功 (≥475): {'是' if np.mean(return_list[-100:]) >= 475 else '否'}")