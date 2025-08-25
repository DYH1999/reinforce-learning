import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import time

# 添加父目录到路径，以便导入 rl_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rl_utils
from SAC_Continuous import SACContinuous

# 解决 matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

class ModelTester:
    """训练好的SAC模型测试器"""
    
    def __init__(self, model_path, env_name='Pendulum-v1'):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # 加载模型
        self.agent = self.load_model(model_path)
        print(f"模型已从 {model_path} 加载完成")
        
    def load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型数据
        checkpoint = torch.load(model_path, map_location=self.device)
        hyperparams = checkpoint['hyperparameters']
        
        print("模型信息:")
        print(f"  环境: {self.env_name}")
        print(f"  状态维度: {hyperparams['state_dim']}")
        print(f"  动作维度: {hyperparams['action_dim']}")
        print(f"  动作边界: {hyperparams['action_bound']}")
        
        if 'training_info' in checkpoint:
            training_info = checkpoint['training_info']
            print(f"  训练回合数: {training_info['num_episodes']}")
            print(f"  最终回报: {training_info['final_return']:.2f}")
            print(f"  平均回报: {training_info['avg_return']:.2f}")
            print(f"  最大回报: {training_info['max_return']:.2f}")
        
        # 创建智能体
        agent = SACContinuous(
            state_dim=hyperparams['state_dim'],
            hidden_dim=hyperparams['hidden_dim'],
            action_dim=hyperparams['action_dim'],
            action_bound=hyperparams['action_bound'],
            actor_lr=3e-4,  # 测试时学习率不重要
            critic_lr=3e-3,
            alpha_lr=3e-4,
            target_entropy=hyperparams['target_entropy'],
            tau=hyperparams['tau'],
            gamma=hyperparams['gamma'],
            device=self.device
        )
        
        # 加载网络参数
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        agent.target_critic_1.load_state_dict(checkpoint['target_critic_1_state_dict'])
        agent.target_critic_2.load_state_dict(checkpoint['target_critic_2_state_dict'])
        agent.log_alpha = checkpoint['log_alpha']
        
        # 设置为评估模式
        agent.actor.eval()
        agent.critic_1.eval()
        agent.critic_2.eval()
        
        return agent
    
    def get_deterministic_action(self, state):
        """获取确定性动作（使用均值，不采样）"""
        state = torch.as_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            x = torch.relu(self.agent.actor.fc1(state))
            mu = self.agent.actor.fc_mu(x)
            # 使用tanh限制动作范围，然后缩放
            action = torch.tanh(mu) * self.agent.actor.action_bound
        return action.squeeze(0).cpu().numpy()
    
    def test_single_episode(self, render=False, deterministic=True):
        """测试单个回合"""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_return = 0
        step_count = 0
        states_history = []
        actions_history = []
        rewards_history = []
        
        done = False
        while not done:
            if render:
                self.env.render()
                time.sleep(0.02)  # 稍微延迟以便观察
            
            # 选择动作
            if deterministic:
                action = self.get_deterministic_action(state)
            else:
                action = self.agent.take_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # 记录历史
            states_history.append(state.copy())
            actions_history.append(action.copy() if isinstance(action, np.ndarray) else action)
            rewards_history.append(reward)
            
            episode_return += reward
            step_count += 1
            state = next_state
        
        return {
            'return': episode_return,
            'steps': step_count,
            'states': states_history,
            'actions': actions_history,
            'rewards': rewards_history
        }
    
    def test_multiple_episodes(self, num_episodes=10, deterministic=True):
        """测试多个回合"""
        print(f"\n开始测试 {num_episodes} 个回合...")
        print(f"动作选择模式: {'确定性' if deterministic else '随机性'}")
        
        results = []
        returns = []
        steps = []
        
        for episode in range(num_episodes):
            result = self.test_single_episode(deterministic=deterministic)
            results.append(result)
            returns.append(result['return'])
            steps.append(result['steps'])
            
            print(f"回合 {episode + 1:2d}: 回报 = {result['return']:8.2f}, 步数 = {result['steps']:3d}")
        
        # 统计信息
        print(f"\n测试统计 ({num_episodes} 回合):")
        print(f"  平均回报: {np.mean(returns):8.2f} ± {np.std(returns):6.2f}")
        print(f"  最大回报: {np.max(returns):8.2f}")
        print(f"  最小回报: {np.min(returns):8.2f}")
        print(f"  平均步数: {np.mean(steps):8.1f} ± {np.std(steps):6.1f}")
        
        return results, returns, steps
    
    def visualize_performance(self, returns_det, returns_stoch=None):
        """可视化性能对比"""
        plt.figure(figsize=(12, 8))
        
        # 回报对比
        plt.subplot(2, 2, 1)
        episodes = range(1, len(returns_det) + 1)
        plt.plot(episodes, returns_det, 'b-o', label='确定性动作', markersize=4)
        if returns_stoch is not None:
            plt.plot(episodes, returns_stoch, 'r-s', label='随机性动作', markersize=4)
        plt.xlabel('回合')
        plt.ylabel('回报')
        plt.title('测试回报对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 回报分布
        plt.subplot(2, 2, 2)
        plt.hist(returns_det, bins=10, alpha=0.7, label='确定性动作', color='blue')
        if returns_stoch is not None:
            plt.hist(returns_stoch, bins=10, alpha=0.7, label='随机性动作', color='red')
        plt.xlabel('回报')
        plt.ylabel('频次')
        plt.title('回报分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 统计对比
        plt.subplot(2, 2, 3)
        stats_det = [np.mean(returns_det), np.std(returns_det), np.max(returns_det), np.min(returns_det)]
        labels = ['均值', '标准差', '最大值', '最小值']
        x_pos = np.arange(len(labels))
        
        plt.bar(x_pos - 0.2, stats_det, 0.4, label='确定性动作', color='blue', alpha=0.7)
        if returns_stoch is not None:
            stats_stoch = [np.mean(returns_stoch), np.std(returns_stoch), np.max(returns_stoch), np.min(returns_stoch)]
            plt.bar(x_pos + 0.2, stats_stoch, 0.4, label='随机性动作', color='red', alpha=0.7)
        
        plt.xlabel('统计指标')
        plt.ylabel('值')
        plt.title('统计指标对比')
        plt.xticks(x_pos, labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 累积回报
        plt.subplot(2, 2, 4)
        cumulative_det = np.cumsum(returns_det)
        plt.plot(episodes, cumulative_det, 'b-', label='确定性动作', linewidth=2)
        if returns_stoch is not None:
            cumulative_stoch = np.cumsum(returns_stoch)
            plt.plot(episodes, cumulative_stoch, 'r-', label='随机性动作', linewidth=2)
        plt.xlabel('回合')
        plt.ylabel('累积回报')
        plt.title('累积回报')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_action_trajectory(self, episode_result):
        """分析单个回合的动作轨迹"""
        states = np.array(episode_result['states'])
        actions = np.array(episode_result['actions'])
        rewards = np.array(episode_result['rewards'])
        
        plt.figure(figsize=(15, 10))
        
        # 状态轨迹
        plt.subplot(2, 3, 1)
        for i in range(states.shape[1]):
            plt.plot(states[:, i], label=f'状态 {i+1}')
        plt.xlabel('时间步')
        plt.ylabel('状态值')
        plt.title('状态轨迹')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 动作轨迹
        plt.subplot(2, 3, 2)
        if len(actions.shape) == 1:
            plt.plot(actions, 'r-', linewidth=2)
        else:
            for i in range(actions.shape[1]):
                plt.plot(actions[:, i], label=f'动作 {i+1}')
            plt.legend()
        plt.xlabel('时间步')
        plt.ylabel('动作值')
        plt.title('动作轨迹')
        plt.grid(True, alpha=0.3)
        
        # 奖励轨迹
        plt.subplot(2, 3, 3)
        plt.plot(rewards, 'g-', linewidth=2)
        plt.xlabel('时间步')
        plt.ylabel('奖励')
        plt.title('奖励轨迹')
        plt.grid(True, alpha=0.3)
        
        # 状态-动作关系（如果是倒立摆）
        if states.shape[1] >= 3:  # 倒立摆有3个状态
            plt.subplot(2, 3, 4)
            plt.scatter(states[:, 0], actions.flatten() if len(actions.shape) > 1 else actions, 
                       c=range(len(states)), cmap='viridis', alpha=0.6)
            plt.xlabel('角度 cos')
            plt.ylabel('动作')
            plt.title('角度-动作关系')
            plt.colorbar(label='时间步')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 3, 5)
            plt.scatter(states[:, 2], actions.flatten() if len(actions.shape) > 1 else actions, 
                       c=range(len(states)), cmap='viridis', alpha=0.6)
            plt.xlabel('角速度')
            plt.ylabel('动作')
            plt.title('角速度-动作关系')
            plt.colorbar(label='时间步')
            plt.grid(True, alpha=0.3)
        
        # 累积奖励
        plt.subplot(2, 3, 6)
        cumulative_rewards = np.cumsum(rewards)
        plt.plot(cumulative_rewards, 'purple', linewidth=2)
        plt.xlabel('时间步')
        plt.ylabel('累积奖励')
        plt.title('累积奖励')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'total_return': episode_result['return'],
            'avg_reward': np.mean(rewards),
            'action_std': np.std(actions),
            'action_range': (np.min(actions), np.max(actions))
        }
    
    def close(self):
        """关闭环境"""
        self.env.close()

def main():
    """主测试函数"""
    model_path = "models/sac_continuous_model.pth"
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请先运行 main.py 训练模型")
        return
    
    # 创建测试器
    tester = ModelTester(model_path)
    
    # 测试确定性动作
    print("\n" + "="*50)
    print("测试确定性动作（使用策略均值）")
    print("="*50)
    results_det, returns_det, steps_det = tester.test_multiple_episodes(
        num_episodes=10, deterministic=True
    )
    
    # 测试随机性动作
    print("\n" + "="*50)
    print("测试随机性动作（策略采样）")
    print("="*50)
    results_stoch, returns_stoch, steps_stoch = tester.test_multiple_episodes(
        num_episodes=10, deterministic=False
    )
    
    # 可视化性能对比
    tester.visualize_performance(returns_det, returns_stoch)
    
    # 分析最佳回合的动作轨迹
    best_episode_idx = np.argmax(returns_det)
    best_episode = results_det[best_episode_idx]
    
    print(f"\n分析最佳回合 (回合 {best_episode_idx + 1}, 回报: {best_episode['return']:.2f}):")
    analysis = tester.analyze_action_trajectory(best_episode)
    print(f"  平均奖励: {analysis['avg_reward']:.3f}")
    print(f"  动作标准差: {analysis['action_std']:.3f}")
    print(f"  动作范围: [{analysis['action_range'][0]:.3f}, {analysis['action_range'][1]:.3f}]")
    
    # 可选：渲染一个回合（需要安装渲染依赖）
    try:
        print("\n渲染一个测试回合...")
        render_result = tester.test_single_episode(render=True, deterministic=True)
        print(f"渲染回合回报: {render_result['return']:.2f}")
    except Exception as e:
        print(f"渲染失败 (这是正常的，如果没有安装渲染依赖): {e}")
    
    tester.close()
    print("\n测试完成!")

if __name__ == "__main__":
    main()