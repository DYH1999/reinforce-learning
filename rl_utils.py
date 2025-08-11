from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    """
    经验回放缓冲区类，用于存储和采样智能体的经验数据
    主要用于off-policy强化学习算法（如DQN、DDPG等）
    """
    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        Args:
            capacity: 缓冲区最大容量，当超过容量时会自动删除最旧的数据
        """
        # collections.deque: 双端队列，支持从两端高效添加和删除元素
        # maxlen参数: 设置队列最大长度，超出时自动删除最旧元素（FIFO先进先出）
        # 相比普通list，deque在头部操作时间复杂度为O(1)，非常适合做缓冲区
        self.buffer = collections.deque(maxlen=capacity) 


    def add(self, state, action, reward, next_state, done): 
        """
        向缓冲区添加一条经验数据
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
        Args:
            batch_size: 采样的批次大小
        Returns:
            tuple: (states, actions, rewards, next_states, dones) 的批次数据
        """
        # random.sample: 从序列中随机选择指定数量的不重复元素
        # 返回一个新列表，不会修改原序列，确保采样的独立性
        transitions = random.sample(self.buffer, batch_size)  # 随机采样
        # zip(*transitions): 解包操作，将[(s1,a1,r1,s1',d1), (s2,a2,r2,s2',d2), ...]
        # 转换为([s1,s2,...], [a1,a2,...], [r1,r2,...], [s1',s2',...], [d1,d2,...])
        state, action, reward, next_state, done = zip(*transitions)  # 解压数据
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
    计算移动平均值，用于平滑奖励曲线
    Args:
        a: 输入数组（通常是奖励序列）
        window_size: 移动窗口大小
    Returns:
        np.array: 平滑后的数组
    """
    # np.insert(a, 0, 0): 在数组开头插入0，便于计算差分
    # np.cumsum: 计算累积和，cumsum[i] = sum(a[0:i+1])
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    # 数组切片操作：利用累积和的差值快速计算移动平均
    # cumulative_sum[window_size:] - cumulative_sum[:-window_size] 得到每个窗口的和
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    # np.arange(1, window_size-1, 2): 生成等差数列 [1, 3, 5, ...]
    r = np.arange(1, window_size-1, 2)
    # [::2]: 每隔2个元素取一个，处理边界情况
    begin = np.cumsum(a[:window_size-1])[::2] / r
    # [::-1]: 数组反转，处理尾部边界
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    # np.concatenate: 沿指定轴连接数组序列
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    """
    训练on-policy智能体的函数（如REINFORCE、Actor-Critic、PPO等）
    on-policy算法使用当前策略收集的数据来更新策略
    
    Args:
        env: 环境对象
        agent: 智能体对象，需要有take_action和update方法
        num_episodes: 总训练回合数
    
    Returns:
        list: 每个回合的累积奖励列表
    """
    return_list = []
    # 将训练分为10个迭代，便于显示进度
    for i in range(10):
        # tqdm: 进度条库，用于显示训练进度和预估剩余时间
        # total: 进度条总长度，desc: 进度条描述文字
        # with语句确保进度条正确关闭和资源释放
        with tqdm(total=int(num_episodes/10), desc='迭代 %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                # 存储一个回合的所有转移数据
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                
                # 执行一个完整回合
                while not done:
                    action = agent.take_action(state)  # 智能体选择动作
                    next_state, reward, done, _ = env.step(action)  # 环境执行动作
                    # 记录转移数据
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                # on-policy算法在每个回合结束后立即更新策略
                agent.update(transition_dict)
                
                # 每10个回合显示一次平均奖励
                if (i_episode+1) % 10 == 0:
                    # pbar.set_postfix: 在进度条右侧显示额外信息（如当前回合数和平均奖励）
                    # np.mean(return_list[-10:]): 计算最近10个回合的平均奖励，用于监控训练效果
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                # pbar.update(1): 进度条前进1步，更新显示
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    """
    训练off-policy智能体的函数（如DQN、DDPG、SAC等）
    off-policy算法可以使用历史数据来更新策略，通过经验回放提高样本效率
    
    Args:
        env: 环境对象
        agent: 智能体对象，需要有take_action和update方法
        num_episodes: 总训练回合数
        replay_buffer: 经验回放缓冲区
        minimal_size: 开始训练前缓冲区的最小数据量
        batch_size: 每次更新使用的批次大小
    
    Returns:
        list: 每个回合的累积奖励列表
    """
    return_list = []
    # 将训练分为10个迭代，便于显示进度
    for i in range(10):
        # tqdm: 进度条库，显示off-policy训练进度
        # 与on-policy不同，off-policy可以在每步都更新，进度显示更频繁
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                
                # 执行一个完整回合
                while not done:
                    action = agent.take_action(state)  # 智能体选择动作
                    next_state, reward, done, _ = env.step(action)  # 环境执行动作
                    # 将经验存储到回放缓冲区
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当缓冲区有足够数据时开始训练
                    if replay_buffer.size() > minimal_size:
                        # 从缓冲区采样一批数据
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        # 使用采样的数据更新智能体
                        agent.update(transition_dict)
                return_list.append(episode_return)
                
                # 每10个回合显示一次平均奖励
                if (i_episode+1) % 10 == 0:
                    # pbar.set_postfix: 在进度条右侧显示训练统计信息
                    # 显示当前回合数和最近10回合的平均奖励，用于监控off-policy训练效果
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                # pbar.update(1): 更新进度条，显示训练进展
                pbar.update(1)
    return return_list



def compute_advantage(gamma, lmbda, td_delta):
    """
    计算GAE（Generalized Advantage Estimation）优势函数
    GAE是一种减少方差的优势函数估计方法，广泛用于Actor-Critic算法
    
    Args:
        gamma: 折扣因子，控制未来奖励的重要性
        lmbda: GAE参数，控制偏差-方差权衡（λ=0时为TD(0)，λ=1时为蒙特卡洛）
        td_delta: TD误差序列，通常为 r + γV(s') - V(s)
    
    Returns:
        torch.tensor: 计算得到的优势函数值序列
    """
    # td_delta.detach(): 从计算图中分离tensor，停止梯度传播
    # .numpy(): 将PyTorch tensor转换为numpy数组，便于后续数值计算
    td_delta = td_delta.detach().numpy()  # 将tensor转换为numpy数组
    advantage_list = []
    advantage = 0.0
    # 从后往前计算优势函数（反向递推）
    # GAE公式: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    for delta in td_delta[::-1]:  # 逆序遍历TD误差
        advantage = gamma * lmbda * advantage + delta  # 递推计算优势函数
        advantage_list.append(advantage)
    advantage_list.reverse()  # 恢复正确的时间顺序
    return torch.tensor(advantage_list, dtype=torch.float)
                