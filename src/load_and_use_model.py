"""
简化的模型加载和使用示例
展示如何加载训练好的SAC模型并在新环境中使用
"""

import torch
import gym
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SAC_Continuous import SACContinuous

def load_trained_model(model_path, device=None):
    """
    加载训练好的SAC模型
    
    Args:
        model_path: 模型文件路径
        device: 计算设备
    
    Returns:
        agent: 加载好的SAC智能体
        hyperparams: 模型超参数
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=device)
    hyperparams = checkpoint['hyperparameters']
    
    # 创建智能体
    agent = SACContinuous(
        state_dim=hyperparams['state_dim'],
        hidden_dim=hyperparams['hidden_dim'],
        action_dim=hyperparams['action_dim'],
        action_bound=hyperparams['action_bound'],
        actor_lr=3e-4,  # 加载时学习率不重要
        critic_lr=3e-3,
        alpha_lr=3e-4,
        target_entropy=hyperparams['target_entropy'],
        tau=hyperparams['tau'],
        gamma=hyperparams['gamma'],
        device=device
    )
    
    # 加载网络参数
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
    agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
    agent.log_alpha = checkpoint['log_alpha']
    
    # 设置为评估模式
    agent.actor.eval()
    
    return agent, hyperparams

def get_optimal_action(agent, state):
    """
    获取最优动作（确定性，使用策略网络的均值）
    
    Args:
        agent: SAC智能体
        state: 当前状态
    
    Returns:
        action: 最优动作
    """
    state_tensor = torch.as_tensor(state, dtype=torch.float, device=agent.device).unsqueeze(0)
    
    with torch.no_grad():
        # 直接使用策略网络的均值输出，不进行随机采样
        x = torch.relu(agent.actor.fc1(state_tensor))
        mu = agent.actor.fc_mu(x)
        # 使用tanh限制并缩放到动作范围
        action = torch.tanh(mu) * agent.actor.action_bound
    
    return action.squeeze(0).cpu().numpy()

def evaluate_model(agent, env, num_episodes=5, render=False):
    """
    评估模型性能
    
    Args:
        agent: SAC智能体
        env: 测试环境
        num_episodes: 测试回合数
        render: 是否渲染
    
    Returns:
        results: 评估结果
    """
    results = {
        'returns': [],
        'episode_lengths': [],
        'success_rate': 0
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_return = 0
        episode_length = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            # 获取最优动作
            action = get_optimal_action(agent, state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
            state = next_state
        
        results['returns'].append(episode_return)
        results['episode_lengths'].append(episode_length)
        
        print(f"回合 {episode + 1}: 回报 = {episode_return:.2f}, 长度 = {episode_length}")
    
    # 计算统计信息
    returns = np.array(results['returns'])
    results['mean_return'] = np.mean(returns)
    results['std_return'] = np.std(returns)
    results['max_return'] = np.max(returns)
    results['min_return'] = np.min(returns)
    
    return results

def main():
    """主函数示例"""
    # 模型路径
    model_path = "models/sac_continuous_model.pth"
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行 main.py 训练模型")
        return
    
    print("加载训练好的SAC模型...")
    
    # 加载模型
    agent, hyperparams = load_trained_model(model_path)
    
    print("模型加载成功!")
    print(f"状态维度: {hyperparams['state_dim']}")
    print(f"动作维度: {hyperparams['action_dim']}")
    print(f"动作边界: {hyperparams['action_bound']}")
    
    # 创建测试环境
    env = gym.make('Pendulum-v1')
    
    print("\n开始评估模型性能...")
    
    # 评估模型
    results = evaluate_model(agent, env, num_episodes=10, render=False)
    
    print(f"\n评估结果:")
    print(f"平均回报: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"最大回报: {results['max_return']:.2f}")
    print(f"最小回报: {results['min_return']:.2f}")
    print(f"平均回合长度: {np.mean(results['episode_lengths']):.1f}")
    
    # 演示单步动作选择
    print(f"\n演示单步动作选择:")
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    print(f"当前状态: {state}")
    
    # 获取随机动作（探索）
    random_action = agent.take_action(state)
    print(f"随机动作: {random_action}")
    
    # 获取最优动作（利用）
    optimal_action = get_optimal_action(agent, state)
    print(f"最优动作: {optimal_action}")
    
    env.close()
    print("\n演示完成!")

if __name__ == "__main__":
    main()