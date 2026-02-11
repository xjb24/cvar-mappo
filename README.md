# CVaR-MAPPO for Cell-Free Massive MIMO Networks

这是一个基于 **R-MAPPO (Recurrent Multi-Agent PPO)** 算法的强化学习项目，用于 **Cell-Free Massive MIMO 网络**的多智能体协同优化。

本项目实现了带 **CVaR (Conditional Value-at-Risk)** 风险约束的 MAPPO 算法变体，用于在不确定环境下的鲁棒网络资源分配。

## 特性

- **算法**: R-MAPPO (Recurrent Proximal Policy Optimization)
- **环境**: Cell-Free Massive MIMO 网络仿真环境
- **风险约束**: 支持 CVaR 风险度量，提供鲁棒性保障
- **共享策略网络**: 所有智能体使用共享的策略网络参数
- **训练入口**: `onpolicy/train_tii.sh`

## 项目结构

```
CVaR-MAPPO-v2/
├── onpolicy/
│   ├── algorithms/
│   │   └── r_mappo/          # R-MAPPO 算法实现
│   ├── runner/
│   │   └── shared/           # 共享策略网络 Runner
│   ├── envs/
│   │   └── cell-free_env/    # Cell-Free 环境实现
│   ├── train_tii.sh          # 训练入口脚本
│   └── train_tii.py          # 训练入口脚本
├── requirements.txt
├── setup.py
└── environment.yaml
```

## 安装

### 环境要求

- Python >= 3.6
- PyTorch >= 1.5.1
- CUDA >= 10.1 (可选，用于 GPU 加速)

### 安装步骤

```bash
# 创建 conda 环境
conda env create -f environment.yaml
conda activate marl

# 或手动创建
conda create -n marl python=3.8
conda activate marl
pip install torch==1.8.0 torchvision==0.9.0

# 安装项目依赖
pip install -e .
```

## 使用方法

### 训练

训练入口为 `train_tii.sh`，配置关键参数：

```bash
cd onpolicy
chmod +x train_tii.sh
./train_tii.sh
```

脚本中的关键配置参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `env` | 环境名称 | `CellFree` |
| `scenario` | 场景名称 | `cell_free` |
| `num_agents` | 智能体数量 | `6` |
| `algo` | 算法类型 | `rmappo` |
| `use_cvar` | 是否使用 CVaR 约束 | `False` |
| `cvar_eta` | CVaR 分位数 | `0.5` |
| `cvar_lambda` | CVaR 正则化系数 | `0.01` |

### 使用 CVaR 约束训练

在 `train_tii.sh` 中设置：
```bash
--use_cvar \
--cvar_eta 0.5 \
--cvar_lambda 0.01 \
```

## 核心组件

### R-MAPPO 算法
- 位置: `onpolicy/algorithms/r_mappo/r_mappo.py`
- 支持循环神经网络 (RNN) 处理部分可观测性
- 集成 CVaR 风险约束优化

### Cell-Free 环境
- 位置: `onpolicy/envs/cell-free_env/`
- 主要文件:
  - `cell_free_env.py`: 环境主实现
  - `config.py`: 环境配置参数
  - `simulation_utils.py`: 信道与速率计算工具

### 共享策略网络 Runner
- 位置: `onpolicy/runner/shared/`
- 基础 Runner: `base_runner.py`
- 支持 MPE、SMAC 等多种环境

## 性能指标

环境返回的关键指标：
- `user_rates`: 用户速率（主要优化目标）
- `episode_reward`: 累积奖励
- `win_rate`: 成功率（如适用）

## 引用

如果本项目对你有帮助，请引用原始 MAPPO 论文：

```bibtex
@inproceedings{yu2022the,
  title={The Surprising Effectiveness of {PPO} in Cooperative Multi-Agent Games},
  author={Chao Yu and Akash Velu and Eugene Vinitsky and Jiaxuan Gao and Yu Wang and Alexandre Bayen and Yi Wu},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```

## 许可证

详见 [LICENSE](LICENSE) 文件。
