import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

# 添加父目录到路径，以便导入 rl_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rl_utils
from SAC_Continuous import SACContinuous

def main():
    # ============ 环境设置 ============
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值

    print(f"环境: {env_name}")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"动作边界: {action_bound}")

    # ============ 设置随机种子 ============
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # ============ 超参数设置 ============
    actor_lr = 3e-4        # 策略网络学习率
    critic_lr = 3e-3       # Q网络学习率
    alpha_lr = 3e-4        # 温度参数学习率
    num_episodes = 100     # 训练回合数
    hidden_dim = 128       # 隐藏层维度
    gamma = 0.99          # 折扣因子
    tau = 0.005           # 软更新参数
    buffer_size = 100000  # 经验回放缓冲区大小
    minimal_size = 1000   # 开始训练前的最小样本数
    batch_size = 64       # 批次大小
    target_entropy = -env.action_space.shape[0]  # 目标熵（负的动作维度）

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"使用设备: {device}")

    # ============ 创建智能体和经验回放缓冲区 ============
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    
    # 定义奖励重塑函数（针对倒立摆环境）
    reward_reshape = lambda r: (r + 8.0) / 8.0
    
    agent = SACContinuous(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        target_entropy=target_entropy,
        tau=tau,
        gamma=gamma,
        device=device,
        reward_reshape=reward_reshape
    )

    print("开始训练...")
    
    # ============ 开始训练 ============
    return_list = rl_utils.train_off_policy_agent(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        replay_buffer=replay_buffer,
        minimal_size=minimal_size,
        batch_size=batch_size
    )

    print("训练完成!")
    
    # ============ 保存训练好的模型 ============
    model_save_path = "models"
    os.makedirs(model_save_path, exist_ok=True)
    
    # 保存模型参数
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_1_state_dict': agent.critic_1.state_dict(),
        'critic_2_state_dict': agent.critic_2.state_dict(),
        'target_critic_1_state_dict': agent.target_critic_1.state_dict(),
        'target_critic_2_state_dict': agent.target_critic_2.state_dict(),
        'log_alpha': agent.log_alpha,
        'hyperparameters': {
            'state_dim': state_dim,
            'hidden_dim': hidden_dim,
            'action_dim': action_dim,
            'action_bound': action_bound,
            'target_entropy': target_entropy,
            'gamma': gamma,
            'tau': tau
        },
        'training_info': {
            'num_episodes': num_episodes,
            'final_return': return_list[-1],
            'avg_return': np.mean(return_list),
            'max_return': np.max(return_list)
        }
    }, os.path.join(model_save_path, 'sac_continuous_model.pth'))
    
    print(f"模型已保存到: {os.path.join(model_save_path, 'sac_continuous_model.pth')}")
    
    # ============ 结果可视化 ============
    episodes_list = list(range(len(return_list)))
    
    # 绘制原始回报曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'SAC Training on {env_name}')
    plt.grid(True)
    
    # 绘制移动平均回报曲线
    plt.subplot(1, 2, 2)
    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Moving Average Returns')
    plt.title(f'SAC Moving Average on {env_name}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # ============ 打印训练统计信息 ============
    print(f"\n训练统计:")
    print(f"最终回报: {return_list[-1]:.2f}")
    print(f"平均回报: {np.mean(return_list):.2f}")
    print(f"最大回报: {np.max(return_list):.2f}")
    print(f"最小回报: {np.min(return_list):.2f}")
    
    # ============ 测试训练好的智能体 ============
    print("\n测试训练好的智能体...")
    test_episodes = 5
    test_returns = []
    
    for episode in range(test_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # 处理新版本gym返回的tuple
        
        episode_return = 0
        done = False
        
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            state = next_state
        
        test_returns.append(episode_return)
        print(f"测试回合 {episode + 1}: {episode_return:.2f}")
    
    print(f"测试平均回报: {np.mean(test_returns):.2f}")
    
    env.close()

if __name__ == "__main__":
    main()