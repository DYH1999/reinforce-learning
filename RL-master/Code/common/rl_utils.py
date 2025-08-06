"""
强化学习通用工具模块
包含经验回放缓冲区、训练函数、优势函数计算等核心组件

主要功能：
1. ReplayBuffer: 经验回放缓冲区，用于off-policy算法的经验存储和采样
2. 训练函数: 支持on-policy和off-policy算法的训练流程
3. 辅助函数: 移动平均、优势函数计算等
"""

from tqdm import tqdm
import numpy as np
import torch
import collections
import random

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')


class ReplayBuffer:
    """
    经验回放缓冲区类 - 强化学习中的核心数据结构
    
    用于存储智能体与环境交互产生的经验数据(s, a, r, s', done)，
    主要用于off-policy强化学习算法（如DQN、DDPG、SAC等）
    
    核心设计理念：
    - 打破数据间的时序相关性：随机采样避免连续样本的相关性
    - 提高样本利用效率：同一经验可以被多次使用
    - 稳定训练过程：缓解在线学习中数据分布变化的问题
    
    Args:
        capacity (int): 缓冲区最大容量，当超过容量时自动删除最旧的数据
    """
    
    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        
        使用collections.deque作为底层数据结构的优势：
        - 双端队列，支持从两端高效添加和删除元素
        - maxlen参数自动实现FIFO（先进先出）机制
        - 相比普通list，在头部操作时间复杂度为O(1)
        
        Args:
            capacity (int): 缓冲区最大容量
        """
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        向缓冲区添加一条经验数据
        
        经验数据格式：(s_t, a_t, r_t, s_{t+1}, done_t)
        - state: 当前状态
        - action: 执行的动作  
        - reward: 获得的即时奖励
        - next_state: 执行动作后的下一个状态
        - done: 是否为终止状态（episode结束标志）
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否为终止状态
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        从缓冲区中随机采样一批经验数据
        
        随机采样的重要性：
        1. 打破时序相关性：避免连续样本间的强相关性影响学习
        2. 提高泛化能力：随机性有助于网络学习更一般的策略
        3. 稳定训练：减少梯度更新的方差
        
        Args:
            batch_size (int): 采样的批次大小
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones) 的批次数据
                - states: 状态批次，numpy数组格式
                - actions: 动作批次，元组格式
                - rewards: 奖励批次，元组格式  
                - next_states: 下一状态批次，numpy数组格式
                - dones: 终止标志批次，元组格式
        """
        # random.sample: 从序列中随机选择指定数量的不重复元素
        # 返回新列表，不修改原序列，确保采样的独立性
        transitions = random.sample(self.buffer, batch_size)
        
        # zip(*transitions): 解包操作，将[(s1,a1,r1,s1',d1), (s2,a2,r2,s2',d2), ...]
        # 转换为([s1,s2,...], [a1,a2,...], [r1,r2,...], [s1',s2',...], [d1,d2,...])
        # 这种转换便于后续的批量处理
        state, action, reward, next_state, done = zip(*transitions)
        
        # 将状态转换为numpy数组，便于后续转换为tensor
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        """
        返回缓冲区当前存储的数据量
        
        Returns:
            int: 缓冲区中的数据条数
        """
        return len(self.buffer)

def moving_average(a, window_size):
    """
    计算移动平均值 - 用于平滑训练曲线
    
    移动平均在强化学习中的作用：
    1. 平滑奖励曲线：减少训练过程中奖励的剧烈波动
    2. 趋势分析：更清晰地观察学习进度和收敛趋势
    3. 性能评估：提供更稳定的性能指标
    
    算法实现细节：
    - 使用累积和技巧提高计算效率，避免重复计算
    - 对边界情况进行特殊处理，确保输出长度与输入一致
    - 分三部分处理：开始部分、中间部分、结束部分
    
    Args:
        a (array-like): 输入数据序列（如每个episode的奖励）
        window_size (int): 移动窗口大小
        
    Returns:
        numpy.ndarray: 移动平均后的数据序列，长度与输入相同
        
    Example:
        >>> rewards = [10, 15, 8, 12, 20, 18, 14]
        >>> smooth_rewards = moving_average(rewards, 3)
        >>> # 返回平滑后的奖励曲线
    """
    # 计算累积和，在开头插入0便于后续计算
    # cumsum([0, a1, a2, a3, ...]) = [0, a1, a1+a2, a1+a2+a3, ...]
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    
    # 中间部分：使用滑动窗口计算移动平均
    # cumulative_sum[window_size:] - cumulative_sum[:-window_size] 
    # 得到每个窗口内元素的和，再除以窗口大小得到平均值
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    
    # 开始部分：窗口不足时的处理
    # 对于前window_size-1个元素，使用逐渐增大的窗口
    r = np.arange(1, window_size-1, 2)  # [1, 3, 5, ...] 奇数序列
    begin = np.cumsum(a[:window_size-1])[::2] / r  # 每隔一个取值并计算平均
    
    # 结束部分：对称处理，确保输出长度一致
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]  # 反向处理后再反转
    
    # 拼接三部分得到完整的移动平均序列
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    """
    训练on-policy强化学习智能体的通用函数
    
    On-policy算法特点：
    1. 策略评估和策略改进使用同一个策略
    2. 必须使用当前策略生成的数据进行学习
    3. 典型算法：REINFORCE、Actor-Critic、PPO、A3C等
    
    训练流程设计：
    - 分10个迭代周期，便于观察训练进度
    - 每个episode收集完整轨迹后进行一次更新
    - 实时显示训练进度和性能指标
    
    Args:
        env: OpenAI Gym环境实例
        agent: 实现了take_action()和update()方法的智能体
        num_episodes (int): 总训练episode数量
        
    Returns:
        list: 每个episode的累积奖励列表，用于分析训练效果
        
    智能体接口要求：
        - take_action(state): 根据状态选择动作
        - update(transition_dict): 根据轨迹数据更新策略
    """
    return_list = []  # 存储每个episode的累积奖励
    
    # 分10个迭代周期进行训练，便于进度监控
    for i in range(10):
        # 使用tqdm显示当前迭代的进度条
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0  # 当前episode的累积奖励
                
                # 存储一个episode的所有转移数据
                # on-policy算法需要完整的episode数据进行更新
                transition_dict = {
                    'states': [],      # 状态序列
                    'actions': [],     # 动作序列  
                    'next_states': [], # 下一状态序列
                    'rewards': [],     # 奖励序列
                    'dones': []        # 终止标志序列
                }
                
                # 重置环境，开始新的episode
                state = env.reset()
                done = False
                
                # 执行一个完整的episode
                while not done:
                    # 处理新版本gym返回的状态格式 (state, info)
                    if len(state) == 2:
                        state = state[0]
                    
                    # 智能体根据当前状态选择动作
                    action = agent.take_action(state)
                    
                    # 执行动作，获取环境反馈
                    next_state, reward, done, _ = env.step(action)[:4]
                    
                    # 存储转移数据 (s_t, a_t, r_t, s_{t+1}, done_t)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    
                    # 更新状态和累积奖励
                    state = next_state
                    episode_return += reward
                
                # episode结束后记录累积奖励
                return_list.append(episode_return)
                
                # on-policy算法：使用刚收集的episode数据更新策略
                # 这是on-policy的核心：必须用当前策略生成的数据学习
                agent.update(transition_dict)
                
                # 每10个episode显示一次平均性能
                if (i_episode+1) % 10 == 0:
                    avg_return = np.mean(return_list[-10:])  # 最近10个episode的平均奖励
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                        'return': '%.3f' % avg_return
                    })
                pbar.update(1)
    
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    """
    训练off-policy强化学习智能体的通用函数
    
    Off-policy算法特点：
    1. 策略评估和策略改进可以使用不同策略的数据
    2. 可以重复利用历史经验数据，提高样本效率
    3. 支持经验回放机制，打破数据相关性
    4. 典型算法：DQN、DDPG、SAC、TD3等
    
    与on-policy的关键区别：
    - 使用经验回放缓冲区存储和重用历史数据
    - 每个时间步都可能进行多次更新
    - 不需要等待完整episode结束就可以学习
    
    Args:
        env: OpenAI Gym环境实例
        agent: 实现了take_action()和update()方法的智能体
        num_episodes (int): 总训练episode数量
        replay_buffer (ReplayBuffer): 经验回放缓冲区
        minimal_size (int): 开始训练前缓冲区的最小数据量
        batch_size (int): 每次更新使用的批次大小
        
    Returns:
        list: 每个episode的累积奖励列表
        
    智能体接口要求：
        - take_action(state): 根据状态选择动作（可能包含探索）
        - update(transition_dict): 根据批次数据更新策略
    """
    return_list = []  # 存储每个episode的累积奖励
    
    # 分10个迭代周期进行训练
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0  # 当前episode的累积奖励
                
                # 重置环境开始新episode
                state = env.reset()
                done = False
                
                # 执行一个完整的episode
                while not done:
                    # 处理新版本gym的状态格式
                    if len(state) == 2:
                        state = state[0]
                    
                    # 智能体选择动作（通常包含探索策略如ε-greedy）
                    action = agent.take_action(state)
                    
                    # 执行动作获取环境反馈
                    next_state, reward, done, _ = env.step(action)[:4]
                    
                    # 将经验存储到回放缓冲区
                    # off-policy的核心：存储经验供后续重复使用
                    replay_buffer.add(state, action, reward, next_state, done)
                    
                    # 更新状态和累积奖励
                    state = next_state
                    episode_return += reward
                    
                    # 当缓冲区有足够数据时开始训练
                    # minimal_size确保有足够的经验多样性
                    if replay_buffer.size() > minimal_size:
                        # 从缓冲区随机采样一批经验
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        
                        # 构造批次数据字典
                        transition_dict = {
                            'states': b_s,      # 状态批次
                            'actions': b_a,     # 动作批次
                            'next_states': b_ns, # 下一状态批次
                            'rewards': b_r,     # 奖励批次
                            'dones': b_d        # 终止标志批次
                        }
                        
                        # off-policy算法：使用随机采样的历史数据更新策略
                        # 这是off-policy的核心：可以用任意策略生成的数据学习
                        agent.update(transition_dict)
                
                # episode结束后记录累积奖励
                return_list.append(episode_return)
                
                # 每10个episode显示一次平均性能
                if (i_episode+1) % 10 == 0:
                    avg_return = np.mean(return_list[-10:])
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                        'return': '%.3f' % avg_return
                    })
                pbar.update(1)
    
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    """
    计算广义优势估计(Generalized Advantage Estimation, GAE)
    
    GAE是强化学习中的重要概念，用于估计动作的相对优势：
    
    理论基础：
    1. 优势函数：A(s,a) = Q(s,a) - V(s)，表示在状态s下选择动作a相比平均水平的优势
    2. TD误差：δ_t = r_t + γV(s_{t+1}) - V(s_t)，是优势函数的无偏估计
    3. GAE结合了偏差和方差的权衡，提供更稳定的优势估计
    
    GAE公式：
    A_t^{GAE(γ,λ)} = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
    
    参数作用：
    - γ (gamma): 折扣因子，控制未来奖励的重要性
    - λ (lambda): GAE参数，控制偏差-方差权衡
      * λ=0: 低方差但高偏差（只使用1步TD误差）
      * λ=1: 低偏差但高方差（使用蒙特卡洛回报）
      * 0<λ<1: 在偏差和方差间取得平衡
    
    算法实现：
    - 从后向前递推计算，避免重复计算
    - 使用递推关系：A_t = δ_t + γλA_{t+1}
    
    Args:
        gamma (float): 折扣因子，通常在[0.9, 0.99]范围内
        lmbda (float): GAE参数，通常在[0.9, 0.98]范围内  
        td_delta (torch.Tensor): TD误差序列，形状为[T]
        
    Returns:
        torch.Tensor: 优势估计序列，形状与td_delta相同
        
    应用场景：
        主要用于Actor-Critic类算法（如PPO、A3C）中的策略梯度计算
    """
    # 将tensor转换为numpy数组进行计算
    # detach()确保不会影响梯度计算
    td_delta = td_delta.detach().numpy()
    
    advantage_list = []  # 存储计算得到的优势值
    advantage = 0.0      # 当前优势值，从后向前递推
    
    # 从序列末尾开始向前计算
    # 递推关系：A_t = δ_t + γλA_{t+1}
    for delta in td_delta[::-1]:  # 反向遍历TD误差序列
        # GAE递推公式的核心计算
        # advantage代表A_{t+1}，delta代表δ_t
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    
    # 由于是反向计算的，需要反转列表得到正确顺序
    advantage_list.reverse()
    
    # 转换回torch tensor供后续使用
    return torch.tensor(advantage_list, dtype=torch.float)
