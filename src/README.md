# SAC (Soft Actor-Critic) 模型训练和测试

这个项目实现了连续动作空间的SAC算法，包含模型训练、保存、加载和测试功能。

## 文件说明

- `SAC_Continuous.py`: SAC算法的核心实现
- `main.py`: 训练脚本，训练并保存模型
- `test_model.py`: 完整的模型测试脚本，包含详细分析和可视化
- `load_and_use_model.py`: 简化的模型加载和使用示例

## 使用步骤

### 1. 训练模型

```bash
cd src
python main.py
```

这将：
- 在Pendulum-v1环境中训练SAC智能体
- 显示训练过程和结果
- 将训练好的模型保存到 `models/sac_continuous_model.pth`

### 2. 测试训练好的模型

#### 方法一：完整测试和分析

```bash
python test_model.py
```

这将：
- 加载训练好的模型
- 测试确定性和随机性动作
- 显示详细的性能统计
- 生成可视化图表
- 分析最佳回合的动作轨迹

#### 方法二：简单加载和使用

```bash
python load_and_use_model.py
```

这将：
- 演示如何加载模型
- 展示如何获取最优动作
- 进行简单的性能评估

## 模型保存格式

保存的模型包含以下信息：
- 所有网络的参数（Actor, Critic1, Critic2, Target networks）
- 温度参数 α
- 超参数配置
- 训练统计信息

## 关键功能

### 确定性动作 vs 随机性动作

- **确定性动作**: 使用策略网络的均值输出，适合最终部署
- **随机性动作**: 使用策略网络采样，保持探索能力

### 性能分析

`test_model.py` 提供了丰富的分析功能：
- 回报对比和分布
- 动作轨迹分析
- 状态-动作关系可视化
- 累积奖励分析

## 在其他项目中使用训练好的模型

```python
from load_and_use_model import load_trained_model, get_optimal_action
import gym

# 加载模型
agent, hyperparams = load_trained_model("models/sac_continuous_model.pth")

# 创建环境
env = gym.make('Pendulum-v1')
state = env.reset()
if isinstance(state, tuple):
    state = state[0]

# 获取最优动作
action = get_optimal_action(agent, state)

# 执行动作
next_state, reward, terminated, truncated, _ = env.step(action)
```

## 环境要求

- Python 3.7+
- PyTorch
- Gym
- NumPy
- Matplotlib
- tqdm

## 注意事项

1. 确保 `rl_utils.py` 在项目根目录或Python路径中
2. 模型文件会保存在 `models/` 目录下
3. 如果要渲染环境，可能需要额外的依赖（如pygame）
4. GPU可用时会自动使用，否则使用CPU

## 自定义使用

你可以轻松修改这些脚本来：
- 使用不同的环境
- 调整超参数
- 添加新的分析功能
- 集成到你的项目中