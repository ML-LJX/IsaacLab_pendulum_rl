原飞书文档链接:
https://sjtu.feishu.cn/wiki/BvhWwCNw6i8k5KkqqYwc8J8Xnmh
本学习文档为个人向教程, 在原有文档的基础上进行了一些改进, 希望对你有所帮助


# 1. 系统准备

参考视频：
https://www.bilibili.com/video/BV1hL411r7p2/?spm_id_from=333.337.search-card.all.click&vd_source=509eab5ce0418134f64744a09dffa26c
安装时选择22.04版本, 兼容性最好


# 2. 环境配置

## 2.1 安装"基建"软件
1. Miniconda
2. VS Code
## 2.2 安装Isaac SiM & Isaac Lab
	以sim4.5.0为例
### 2.2.1 创建conda环境
    Isaac Sim 4.5.0 **必须**使用 Python 3.10。
```Bash
conda create -n env_IsaacSim450 python=3.10 -y
conda activate env_IsaacSim450
```
### 2.2.2 安装Isaac Sim 4.5.0
	使用 NVIDIA 官方 PyPI 源进行 Pip 安装。
```Bash
# 升级 pip 以避免依赖解析错误
pip install --upgrade pip

# 安装 Sim 4.5 (核心步骤)# --extra-index-url 是必须的
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
```
### 2.2.3 安装Isaac Lab (源码安装)
```Bash
# 找个工作目录
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```
**注意**
`git clone https://github.com/isaac-sim/IsaacLab.git`这行命令会自动克隆官方lab的最新版本，下载前先确认一下自己需要什么版本，老版本就去github仓库自己下之前的压缩包。当然就这个单&双&三摆而言不需要考虑这些


# 3. 单摆

## 3.0 跑通官方训练代码
	lab有很多库的代码，这里以运行rsl_rl为例
```Plain
#开始训练，当然你可以不加headless看看训练过程，发现会逐步进入稳定状态。
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-Direct-v0 --headless
#训练结束后，可以play看看效果
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Cartpole-Direct-v0
#可以在play时按shift和左键拖拽机器人
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Cartpole-Direct-v0 --device cpu 
```
**注意**
Isaac Sim启动时经常出现"假死"现象, 一直等待即可

## 3.1 单摆代码魔改

### 任务1: 起摆
#### i. 修改初始状态
```Python
class CartpoleEnvCfg(DirectRLEnvCfg):
	# ... 原有代码 ...
	# [修改点] 将观测空间从 4 改为 6 (因为后续引入了sin,cos和位置误差)  
	observation_space = 6
	
	# [修改点] 将初始角度范围设置为自然下垂状态
    initial_pole_angle_range = [1 - 0.08, 1 + 0.08] 
```
**修改点1: 增加观测空间维数, 将初始角度范围设置为(1 $\pm$ 0.08)$\pi$**
#### ii. 放宽终止条件
```Python
def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
	# ... 原有代码 ...
    out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1) 
    
    # [修改点] 注释掉关于杆子角度 > π/2 就 done 的条件，允许杆子大范围摆动以完成起摆 
    # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
```
**修改点2: 取消$\pi /2$的角度限制**
#### iii. 连续角度的观测处理
```python
def _get_observations(self) -> dict:
    pole_pos = self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1)
    obs = torch.cat( 
	    ( 
		    # [修改点] 把原生角度替换为sin和cos
		    torch.sin(pole_pos), 
		    torch.cos(pole_pos), 
		    # ... 原有代码 ... 
		), 
		dim=-1,
	)
```
**修改点3: 把原生角度替换为sin和cos, 避免角度在-$\pi$到$\pi$之间突变带来的数值不连续性, 帮助神经网络更好地理解旋转**
#### iv. 奖励函数的角度包裹
```python
def compute_rewards(
	# ... 原有代码 ...
):
	# [修改点] 角度归一化
	pole_pos_wrapped = (pole_pos + torch.pi) % (2.0 * torch.pi) - torch.pi
	rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos_wrapped).unsqueeze(dim=1), dim=-1)
```
**修改点4: 进行角度归一化, 将连续角度强制映射到$[- \pi , \pi]$**

### 任务2: 抗扰动
#### i. 物理外力扰动
```Python
def _pre_physics_step(self, actions: torch.Tensor) -> None:
    # ... 原有代码 ...
    # [修改点] 引入随机物理扰动
    env_ids_to_kick = torch.rand(self.num_envs, device=self.device) < 0.01
    if torch.any(env_ids_to_kick):
	    kick_force = (torch.rand_like(self.actions[env_ids_to_kick]) - 0.5) * 200.0
	    self.actions[env_ids_to_kick] += kick_force
```
**修改点1: 每个环境在每个step有1%的概率会受到一个随机方向最大值为100N的力**
#### ii. 传感器噪声扰动
```Python
def _get_observations(self) -> dict: 
    # ... 原有代码 ...
    # [修改点] 引入观测噪声
    noise = (torch.rand_like(obs) - 0.5) * 0.1 
    obs = obs + noise
```
**修改点2: 给整个观测向量加入 $[-0.05, 0.05]$ 之间的均匀分布噪声, 模拟真实世界传感器的读取误差**

### 任务3: 定点平衡
#### i. 引入目标位置
```python
def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
	# ... 原有代码 ...
	# [修改点] 新增目标位置张量
	self.target_pos = torch.ones(self.num_envs, 1, device=self.device) * 1.0
```
**修改点1: 新增self.target_pos, 硬编码为1.0**
#### ii. 引入位置误差
```python
def _get_observations(self) -> dict:
	# ... 原有代码 ...
	obs = torch.cat( 
		(
			 # ... 原有代码 ... 
			 # [修改点] 把小车当前位置与目标位置的绝对误差放进观测空间
			 (self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1) - self.target_pos) 
		), 
		dim=-1,
	) 
```
**修改点2: 把位置误差放进观测空间**
#### iii. 距离惩罚奖励
```python
def compute_rewards(
	# ... 原有代码 ...
):
	# ... 原有代码 ...
	# [修改点] 增加定点约束
	target_pos = 1.0
    pos_error = torch.abs(cart_pos - target_pos)
    # [修改点] 使用线性惩罚项
    rew_position = -1.5 * pos_error 
    total_reward += rew_position
```
**修改点3: 使用线性惩罚项进行空间约束**


# 4. 双摆

## 4.0 Isaac Sim GUI基础操作
1. 在你的conda环境下, cd到相应的文件夹后, 输入./isaaclab.sh -s即可进入GUI图形化编辑器界面
2. 在GUI的左下角Isaac Sim Assets [Beta]中输入double搜索, 找到双摆usd文件, 右键add at current selection, 即可打开双摆usd文件
3. 打开双摆文件后, 按住alt+鼠标左键可拖动视角, 按住鼠标右键+w/a/s/d可以移动观察位置
4. 点击GUI的左侧播放键, 应该能看到双摆停在原地不动, 然后按住shift给任意杆子一个力, 就能看到双摆受力后自由落下的情况
5. GUI的右上角Stage显示了目前的文件
6. GUI的右下角Property面板显示了文件的各种属性, 可以自己调一调玩玩

## 4.1 双摆代码魔改
**注意**
IsaacLab官方的双摆环境代码使用的是MAPPO多智能体算法, 原飞书文档的教程中也是在MAPPO算法基础上进行的修改, 且对第二根杆子施加了力矩, 但是对于双摆&三摆这种混沌体系, 单智能体PPO算法效果显著强于MAPPO, 笔者的双摆&三摆代码均将原文档中的MAPPO改为了PPO, 并取消了对第二根杆子的力矩, 故相较于原文档的代码魔改部分, 笔者进行了更多的修改, 毕竟这是一份$*个人向*$学习文档. 希望能对你有所帮助
### i. 配置类与底层架构重构 (MARL -> RL)
```python
# [修改点] 引入的基类从 DirectMARLEnv 变更为 DirectRLEnv
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg

# [修改点] 继承类从 DirectMARLEnvCfg 变更为 DirectRLEnvCfg
class CartDoublePendulumEnvCfg(DirectRLEnvCfg):
    
    # [修改点] 新增任务模式开关与目标状态记录
    task_mode = "up_up" 
    target_state = [0.0, 0.0, 0.0] 
    
    # ... 原有代码 ...
    
    # [修改点] 删除多智能体相关的 possible_agents = ["cart", "pendulum"]
    # [修改点] 将字典形式的 action_spaces 和 observation_spaces 拍扁为单智能体数字
    action_space = 1
    observation_space = 7
    # 原代码为 state_space = -1，修改为 0
    state_space = 0
    
    # ... 原有代码 ...
    
    # [修改点] 删除第二根杆子的驱动力配置 (纯欠驱动)
    # 删除了 pendulum_action_scale = 50.0  # [Nm]
```
**修改点1: 将底层架构从MARL改为RL, 移除了第二根杆子的电机配置, 同时新增了task_mode变量**
### ii. 动作处理与驱动逻辑
```python
# [修改点] 继承的主环境类也同步修改为 DirectRLEnv
class CartDoublePendulumEnv(DirectRLEnv):
    # ...
    
    # [修改点] 输入参数 actions 从字典 dict[str, torch.Tensor] 改为纯张量 torch.Tensor
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # [修改点] 单智能体环境需要 clone() 防止张量内存污染
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # [修改点] 仅保留小车 (cart) 的动作计算，切片取第一维
        cart_effort = self.actions[:, 0:1] * self.cfg.cart_action_scale
        
        # 仅对小车关节施加力矩
        self.robot.set_joint_effort_target(cart_effort, joint_ids=self._cart_dof_idx)
        
        # [修改点] 删除了对第二根杆子 (pendulum) 施加力矩的代码
        # 删除了 self.robot.set_joint_effort_target(...)
```
**修改点2: 剥夺连杆的主动发力能力, 动作张量直接作为一维张量接收, 并仅仅应用给底盘小车。**
### iii. 观测空间的整合与特征工程
```python
    # [修改点] 返回值类型从 dict[str, torch.Tensor] 改为单智能体规范的 dict
    def _get_observations(self) -> dict:
        
        pole_joint_pos = normalize_angle(self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1))
        pendulum_joint_pos = normalize_angle(self.joint_pos[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1))
        
        # [修改点] 取消 "cart" 和 "pendulum" 的分块字典，将所有观测值拼接为一个长度为 7 的一维张量
        obs = torch.cat(
            (
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                pole_joint_pos,
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                # [修改点] 加入了绝对角度特征 (pole + pendulum)
                pole_joint_pos + pendulum_joint_pos, 
                # [修改点] 加入了相对角度特征 (pendulum)
                pendulum_joint_pos,                  
                self.joint_vel[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return {"policy": obs}
```
**修改点3: 将观测信息聚合成一个空间维数为7的综合观测, 并在神经网络的输入特征中给出第二根杆子的绝对角度和相对角度**
### iv. 改进奖励函数
```python
    # [修改点] 返回值类型从 dict[str, torch.Tensor] 改为单张量 torch.Tensor
    def _get_rewards(self) -> torch.Tensor:
        # [修改点] 根据 task_mode 动态下发目标角度 (Target Angles)
        if self.cfg.task_mode == "up_up":
            target_p1, target_p2 = 0.0, 0.0
        elif self.cfg.task_mode == "up_down":
            target_p1, target_p2 = 0.0, 3.14159
        elif self.cfg.task_mode == "down_up":
            target_p1, target_p2 = 3.14159, 0.0
            
        total_reward = compute_rewards(
            # ... 传入目标值等参数 ...
        )
        return total_reward

def compute_rewards(
    # ... 原有参数 ...
    # [修改点] 新增目标角度参数
    target_p1: float,
    target_p2: float,
):
    # ... 原有代码 ...
    
    # [修改点] 废弃了原版暴力的二次惩罚 (rew_pole_pos = ... * torch.square(pole_pos))
    # [修改点] 引入误差计算，并换成了基于高斯分布 (指数形式) 的平滑奖励
    error_p1 = normalize_angle(pole_pos - target_p1)
    rew_pole_up = torch.exp(-torch.square(error_p1) / 0.25)
    
    error_p2 = normalize_angle(pole_pos + pendulum_pos - target_p2)
    rew_pendulum_up = torch.exp(-torch.square(error_p2) / 0.25)
    
    # [修改点] 返回值不再是分发给 "cart" 和 "pendulum" 的字典，而是加权求和后的标量奖励
    total_reward = rew_alive + rew_termination + (10.0 * rew_pole_up) + (10.0 * rew_pendulum_up) + rew_cart_vel + rew_pole_vel + rew_pendulum_vel
    
    return total_reward
```
**修改点4: 将奖励函数改为高斯平滑奖励, 并引入动态目标以支持多任务训练**
### v. 动态终止条件
```python
    # [修改点] 返回值从 tuple[dict, dict] 变为纯张量元组 tuple[torch.Tensor, torch.Tensor]
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        
        # [修改点] 动态获取目标角度（为了适配 down_up 等任务）
        target_p1 = 3.14159 if self.cfg.task_mode == "down_up" else 0.0
        
        # [修改点] 失败条件不再是固定超过 π/2，而是“与当前目标的误差”超过 π/2
        error_p1 = normalize_angle(self.joint_pos[:, self._pole_dof_idx] - target_p1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(error_p1) > math.pi / 2, dim=1)   
            
        # [修改点] 直接返回张量，不再包装成 {agent: out_of_bounds} 字典
        return out_of_bounds, time_out
```
**修改点5: 动态判定失败条件, 为起摆等任务保留了合理的探索空间**
### vi. 重置逻辑与防污染处理
```python
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # ...
        # [修改点] 关键防坑：使用 .clone() 读取默认状态，防止原版直接修改造成的内存污染 (Buffer Pollution)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        
        # [修改点] 废弃了官方在 Config 中写死的随机范围，改为根据 task_mode 动态设定初始范围
        if self.cfg.task_mode == "up_up":
            range_1 = [-0.1, 0.1]
            range_2 = [-0.1, 0.1]
        elif self.cfg.task_mode == "up_down":
            range_1 = [-0.1, 0.1]
            range_2 = [3.14 - 0.1, 3.14 + 0.1]
        elif self.cfg.task_mode == "down_up":
            range_1 = [3.14 - 0.1, 3.14 + 0.1]
            range_2 = [3.14 - 0.1, 3.14 + 0.1]
        
        # [修改点] 使用动态生成的 range 覆盖原来的固定 self.cfg.initial_pole_angle_range
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            range_1[0], range_1[1], joint_pos[:, self._pole_dof_idx].shape, joint_pos.device
        )
        joint_pos[:, self._pendulum_dof_idx] += sample_uniform(
            range_2[0], range_2[1], joint_pos[:, self._pendulum_dof_idx].shape, joint_pos.device
        )
        
        # [修改点] 清理了官方原代码中冗余的 joint_vel 和 default_root_state 的重复获取代码
        # ... 写回模拟器 ...
```
**修改点6: 增加.clone()操作以杜绝底层张量内存污染，并根据任务模式动态下发不同的初始角度范围**

## 4.2 修改注册表
	该__init__注册表在cart_double_pendulum文件夹下
```python
gym.register(
    # ... 原有代码 ...
    kwargs={
        # ... 原有代码 ...
        # 新增检索路径
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartDoublePendulumPPORunnerCfg",
    },
)
```
**新增检索路径, 读取/agents/rsl_rl_ppo_cfg文件中的CartDoublePendulumPPORunnerCfg**

## 4.3 新建配置文件
	在/agents文件夹新建rsl_rl_ppo_cfg.py文件,对应__init__注册表中的内容
```python
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class CartDoublePendulumPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000 
    save_interval = 50
    experiment_name = "cart_double_pendulum"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128], # 适合低维环境的轻量级网络
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        learning_rate=3e-4, 
        num_learning_epochs=5,
        num_mini_batches=4,
        gamma=0.99,
        lam=0.95,
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```
**可以自行调整配置文件中的参数, 以达到更好的学习效果**

**小tips:**
train的时候看到单次iteration中的mean_reward基本收敛了就可以直接ctrl+C去play看效果了


# 5. 三摆

## 5.0 创建三摆usd文件
1. 进入Isaac Sim GUI界面, 打开双摆usd文件, 另存为cart_triple_pendulum.usd文件
2. 在右侧Stage界面, 点击pendulum整根杆子, ctrl+D复制, 重命名为pendulum_3
3. 在右侧Stage界面, 点击pole/pole_to_pendulum这个节点, ctrl+D复制, 重命名为pole_to_pendulum_3, 放入/pendulum
4. 拖拽复制的新杆子和节点, 使之成为三摆, 点GUI左侧播放键能自由下落即可. 建议直接在Property面板里改数据, 更容易接上节点和杆子

## 5.1 三摆代码魔改 
**注意**
跟双摆类似, 笔者依然使用单智能体PPO进行训练, 不设置对第二第三跟杆子的力矩. 具体代码对已魔改后的双摆代码进行再修改

### i. 三摆资产导入
```python
# [修改点] 引入了 os 模块，用于获取绝对路径
import os

# [修改点] 获取当前脚本目录，用于动态绑定本地 USD 文件
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# [修改点] 通过直接复制双摆的配置来“移花接木”，而不从头编写三摆的 ArticulationCfg
_TRIPLE_PENDULUM_CFG = CART_DOUBLE_PENDULUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
# [修改点] 强制将底层加载的 USD 路径修改为你本地制作的三摆模型文件
_TRIPLE_PENDULUM_CFG.spawn.usd_path = f"{_CURRENT_DIR}/cart_triple_pendulum.usd"
```
**修改点1: 利用本地绝对路径动态替换USD资产, 通过拷贝并覆盖双摆配置, 实现接入自定义的三摆物理模型**
### ii. 环境配置
```python
class CartTriplePendulumEnvCfg(DirectRLEnvCfg):
    
    # [修改点] 任务模式更改为代表三杆全上的 "up_up_up"
    task_mode = "up_up_up" 
    # [修改点] 目标状态扩展为 4 维 (小车+三根杆)
    target_state = [0.0, 0.0, 0.0, 0.0] 
    
    # [修改点] 观测空间 (observation_space) 从 7 增加到 10 
    # (新增了第三根杆的绝对角度、相对角度和角速度)
    observation_space = 10
    
    # [修改点] 应用修改过 usd_path 的三摆配置
    robot_cfg: ArticulationCfg = _TRIPLE_PENDULUM_CFG
    
    # [修改点] 新增第三根杆子的 DOF (自由度) 名称绑定
    pendulum3_dof_name = "pole_to_pendulum_3"

    # [修改点] 显式声明了三根杆子的初始随机角度范围（虽然在 reset 中会覆盖，但在配置中进行了占位声明）
    initial_pole_angle_range = [-0.25, 0.25]  
    initial_pendulum_angle_range = [-0.25, 0.25]  
    initial_pendulum3_angle_range = [-0.25, 0.25]

    # [修改点] 增加了第三根杆子的位置和速度惩罚缩放系数
    rew_scale_pendulum3_pos = -1.0
    rew_scale_pendulum3_vel = -0.01
```
**修改点2: 扩展观测空间维数, 将配置类的底层模型指向新对象, 增加名称映射与奖励缩放因子**
### iii. 关节初始化与观测空间重构
```python
class CartTriplePendulumEnv(DirectRLEnv):
    def __init__(self, ...):
        # ...
        # [修改点] 在初始化时获取第三根杆的关节索引
        self._pendulum3_dof_idx, _ = self.robot.find_joints(self.cfg.pendulum3_dof_name)

    def _get_observations(self) -> dict:
        # ...
        # [修改点] 提取并标准化第三根杆的局部关节角度
        pendulum3_joint_pos = normalize_angle(self.joint_pos[:, self._pendulum3_dof_idx[0]].unsqueeze(dim=1))
        
        obs = torch.cat(
            (
                # ... 前4个维度不变 (小车pos/vel, 杆1pos/vel) ...
                
                # [修改点] 修正：双摆中直接相加，三摆中加上了 normalize_angle() 以保证复合角度始终在 [-π, π] 内
                normalize_angle(pole_joint_pos + pendulum_joint_pos), 
                pendulum_joint_pos,                                    
                self.joint_vel[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1),
                
                # [修改点] 新增维度 1：第三根杆的绝对角度 (极坐标系累加：杆1 + 杆2 + 杆3)
                normalize_angle(pole_joint_pos + pendulum_joint_pos + pendulum3_joint_pos), 
                # [修改点] 新增维度 2：第三根杆的相对角度
                pendulum3_joint_pos,                                                        
                # [修改点] 新增维度 3：第三根杆的角速度
                self.joint_vel[:, self._pendulum3_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return {"policy": obs}
```
**修改点3: 将第三根杆的参数放入观测空间, 使用normalize_angle防止组合后的绝对角度溢出**
### iv. 动态终止条件与重置逻辑
```python
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # [修改点] 删除了对 cfg.task_mode == "up_up" 的判断，因为三摆目前仅聚焦全直立任务
        # target_p1 硬编码为 0.0
        target_p1 = 0.0
        error_p1 = normalize_angle(self.joint_pos[:, self._pole_dof_idx] - target_p1)
        # [修改点] 终止条件仍然只挂钩第一根杆(pole)，允许第二、第三根杆自由探索而不触发 done
        out_of_bounds = out_of_bounds | torch.any(torch.abs(error_p1) > math.pi / 2, dim=1)   
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # ...
        # [修改点] 简化了模式判断，移除了双摆中 up_down, down_up 的剧烈摆动区间，统一改为 [-0.1, 0.1] 微小扰动
        if self.cfg.task_mode == "up_up_up":
            range_1, range_2, range_3 = [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]
        else:
            range_1, range_2, range_3 = [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]
        
        # ...
        # [修改点] 增加对第三根杆初始位置的均匀分布采样扰动注入
        joint_pos[:, self._pendulum3_dof_idx] += sample_uniform(
            range_3[0], range_3[1], joint_pos[:, self._pendulum3_dof_idx].shape, joint_pos.device
        )
```
**修改点4: 将任务固定为三摆up_up_up, 初始状态由大幅度随机抛甩改为小幅度平衡扰动, 设定只有第一根杆倾角过大才判负。**
### v. 奖励函数设计与分层权重
```python
    def _get_rewards(self) -> torch.Tensor:
        # [修改点] 目标从 2 个变量扩展为 3 个变量 (target_p3)
        if self.cfg.task_mode == "up_up_up":
            target_p1, target_p2, target_p3 = 0.0, 0.0, 0.0
        else:
            target_p1, target_p2, target_p3 = 0.0, 0.0, 0.0
            
        total_reward = compute_rewards(
            # ...
            # [修改点] 传参增加：第三杆的位置缩放、速度缩放、当前pos/vel、目标p3
            self.cfg.rew_scale_pendulum3_pos, 
            self.cfg.rew_scale_pendulum3_vel, 
            # ...
            normalize_angle(self.joint_pos[:, self._pendulum3_dof_idx[0]]), 
            self.joint_vel[:, self._pendulum3_dof_idx[0]],                  
            # ...
            target_p3, 
        )
        return total_reward

@torch.jit.script
def compute_rewards(
    # ... [修改点] 方法签名同步增加了 5 个与第三摆相关的入参
):
    # ...
    # [修改点] 新增对第三根杆的高斯平滑奖励计算，同样基于“绝对角度”误差
    error_p3 = normalize_angle(pole_pos + pendulum_pos + pendulum3_pos - target_p3)
    rew_pendulum3_up = torch.exp(-torch.square(error_p3) / 0.25)
    
    # [修改点] 新增对第三根杆的角速度惩罚
    rew_pendulum3_vel = rew_scale_pendulum3_vel * torch.sum(torch.abs(pendulum3_vel).unsqueeze(dim=1), dim=-1)

    # [修改点] 核心权重变更：引入倒立摆课程式层级权重 (Hierarchical Weights)
    # 双摆比例 -> Pole: 10.0, Pendulum: 10.0
    # 三摆比例 -> Pole: 15.0, Pendulum: 10.0, Pendulum3: 5.0
    total_reward = (
        rew_alive + rew_termination 
        + (15.0 * rew_pole_up) 
        + (10.0 * rew_pendulum_up) 
        + (5.0 * rew_pendulum3_up) 
        + rew_cart_vel + rew_pole_vel + rew_pendulum_vel + rew_pendulum3_vel
    )
    return total_reward
```
**修改点5: 高斯平滑奖励拓展到第三摆, 使用层次化的奖励权重, 引导算法“自下而上”地学习稳定性**

## 5.2 修改注册表
	该__init__注册表在cart_double_pendulum文件夹下
```python
gym.register(
    # ... 原有代码 ...
    # 新增三摆路径
    gym.register(
    id="Isaac-Cart-Triple-Pendulum-Direct-v0", 
    entry_point=f"{__name__}.cart_triple_pendulum_env:CartTriplePendulumEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_triple_pendulum_env:CartTriplePendulumEnvCfg", 
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_triple_ppo_cfg:CartTriplePendulumPPORunnerCfg",
    },
)
)
```
**新增三摆文件检索路径**

## 5.3 新建配置文件
	在/agents文件夹新建rsl_rl_triple_ppo_cfg.py文件,对应__init__注册表中的内容
```python
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class CartTriplePendulumPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 120 # 步数增加到5倍
    max_iterations = 2000 
    save_interval = 50
    experiment_name = "cart_triple_pendulum"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128], # 隐藏层增加一维
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        learning_rate=3e-4, 
        num_learning_epochs=5,
        num_mini_batches=4,
        gamma=0.99,
        lam=0.95,
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```
