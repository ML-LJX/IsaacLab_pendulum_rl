# IsaacLab-Pendulum-RL: 单/双/三级倒立摆强化学习环境

本项目基于 **NVIDIA IsaacLab** 构建，包含单级、双级和三级倒立摆（Cartpole）的强化学习自定义环境。针对多级倒立摆这一经典的混沌系统，本项目对官方基线环境进行了深度重构与优化，旨在提供一个更高效、更稳定的连续控制算法验证平台。

## ✨ 核心特性与优化设计

相比于官方默认配置，本项目进行了以下关键重构：

* **单智能体 PPO 架构 (RL vs MARL)**：摒弃了官方双摆环境中的 MAPPO 多智能体设定，剥夺了连杆的主动发力能力（纯欠驱动），采用单智能体 PPO 算法统一控制底盘小车，显著提升了双摆与三摆任务的收敛速度和平衡稳定性。
* **观测空间特征工程**：引入 $\sin$ 和 $\cos$ 连续角度表征，避免了角度在极端位置突变带来的数值不连续性；并显式地将多级连杆的“绝对角度”与“相对角度”同时喂给神经网络。
* **高斯平滑奖励与动态终止机制**：废弃了原版暴力的二次惩罚，采用基于高斯分布（指数形式）的平滑奖励；引入多任务动态目标，放宽了起摆阶段的终止条件，给予智能体更充足的探索空间。
* **层级化课程学习权重**：在三级倒立摆任务中，设计了自下而上的奖励权重分层（Pole: 15.0 > Pendulum: 10.0 > Pendulum3: 5.0），引导算法逐步攻克多级平衡难题。
* **物理与传感器噪声注入**：在训练中引入随机物理扰动（Kick Force）和观测噪声，提升策略的鲁棒性。

## ⚙️ 环境依赖与安装

本项目依赖 **Isaac Sim 4.5.0** 与 **IsaacLab**。推荐使用 Python 3.10 环境。

**1. 创建并激活 Conda 环境**
```bash
conda create -n isaaclab_env python=3.10 -y
conda activate isaaclab_env
````

**2. 安装 Isaac Sim 4.5.0**

Bash

```
pip install --upgrade pip
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url [https://pypi.nvidia.com](https://pypi.nvidia.com)
```

**3. 安装 IsaacLab**

Bash

```
git clone [https://github.com/isaac-sim/IsaacLab.git](https://github.com/isaac-sim/IsaacLab.git)
cd IsaacLab
./isaaclab.sh --install
```

## 🚀 快速开始

本项目作为 IsaacLab 的外部扩展（Out-of-Tree Extension）运行。克隆本仓库后，可直接通过内置脚本启动训练。

```bash
# 1. 克隆本项目并进入代码目录
git clone [https://github.com/你的用户名/IsaacLab_pendulum_rl.git](https://github.com/你的用户名/IsaacLab_pendulum_rl.git)
cd IsaacLab_pendulum_rl

# 2. 激活之前配置好的 conda 环境
conda activate isaaclab_env
```

### 训练模型

为了获得更快的训练速度，建议在无头模式（`--headless`）下进行训练：

```bash
# train单摆
python scripts/train.py --task=Isaac-Cartpole-Direct-v0 --headless

# train双摆
python scripts/train.py --task=Isaac-Cart-Double-Pendulum-Direct-v0 --headless

# train三摆
python scripts/train.py --task=Isaac-Cart-Triple-Pendulum-Direct-v0 --headless
```

### 渲染与测试

当终端输出的 `mean_reward` 基本收敛时，可以按 `Ctrl+C` 停止训练，并运行以下命令在 GUI 中查看效果（支持鼠标左键+Shift拖拽施加外力）：

```bash
# play单摆
python scripts/play.py --task=Isaac-Cartpole-Direct-v0 

# play双摆
python scripts/play.py --task=Isaac-Cart-Double-Pendulum-Direct-v0 

# play三摆
python scripts/play.py --task=Isaac-Cart-Triple-Pendulum-Direct-v0
```

## 📂 仓库结构

Plaintext

```
├── .gitignore                   # Git 忽略规则配置
├── LICENSE                      # 开源许可证 (MIT)
├── README.md                    # 项目说明文档
├── scripts/
│   ├── cli_args.py              # 命令行参数解析辅助脚本
│   ├── train.py                 # 训练入口脚本
│   └── play.py                  # 测试与渲染脚本
└── pendulum_envs/               # 核心环境源码
    ├── __init__.py              # 注册表，包含所有自定义环境的注册逻辑
    ├── cartpole/                # 单摆环境与 PPO 配置
    └── cart_double_pendulum/    # 双摆/三摆环境、PPO 配置及 USD 资产
```
