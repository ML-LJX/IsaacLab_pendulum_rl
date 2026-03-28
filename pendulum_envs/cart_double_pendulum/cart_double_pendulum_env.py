# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG


@configclass
class CartDoublePendulumEnvCfg(DirectRLEnvCfg):
    
    task_mode = "up_up" 
    target_state = [0.0, 0.0, 0.0] 
    
    # env
    decimation = 2
    episode_length_s = 5.0
    
    action_space = 1
    observation_space = 7
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CART_DOUBLE_PENDULUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    pendulum_dof_name = "pole_to_pendulum"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]

    # action scales
    cart_action_scale = 100.0  # [N]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_cart_pos = 0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_pos = -1.0
    rew_scale_pole_vel = -0.01
    rew_scale_pendulum_pos = -1.0
    rew_scale_pendulum_vel = -0.01


class CartDoublePendulumEnv(DirectRLEnv):
    cfg: CartDoublePendulumEnvCfg

    def __init__(self, cfg: CartDoublePendulumEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        self._pendulum_dof_idx, _ = self.robot.find_joints(self.cfg.pendulum_dof_name)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        cart_effort = self.actions[:, 0:1] * self.cfg.cart_action_scale
        
        self.robot.set_joint_effort_target(cart_effort, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        
        pole_joint_pos = normalize_angle(self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1))
        pendulum_joint_pos = normalize_angle(self.joint_pos[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1))
        obs = torch.cat(
            (
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                pole_joint_pos,
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                pole_joint_pos + pendulum_joint_pos, # 绝对角度
                pendulum_joint_pos,                  # 相对角度
                self.joint_vel[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.task_mode == "up_up":
            target_p1, target_p2 = 0.0, 0.0
        elif self.cfg.task_mode == "up_down":
            target_p1, target_p2 = 0.0, 3.14159
        elif self.cfg.task_mode == "down_up":
            target_p1, target_p2 = 3.14159, 0.0
            
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_cart_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_pole_vel,
            self.cfg.rew_scale_pendulum_pos,
            self.cfg.rew_scale_pendulum_vel,
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            normalize_angle(self.joint_pos[:, self._pole_dof_idx[0]]),
            self.joint_vel[:, self._pole_dof_idx[0]],
            normalize_angle(self.joint_pos[:, self._pendulum_dof_idx[0]]),
            self.joint_vel[:, self._pendulum_dof_idx[0]],
            self.reset_terminated,
            target_p1, 
            target_p2,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        
        target_p1 = 3.14159 if self.cfg.task_mode == "down_up" else 0.0
        
        error_p1 = normalize_angle(self.joint_pos[:, self._pole_dof_idx] - target_p1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(error_p1) > math.pi / 2, dim=1)   
            
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        
        # 1. 克隆初始状态
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        
        # 2. 赋予随机角度
        
        # 强制初始状态为down-down，必须甩动
        # range_1 = [3.14 - 0.1, 3.14 + 0.1]
        # range_2 = [-0.1, 0.1]
        
        # 初始状态为平衡状态
        if self.cfg.task_mode == "up_up":
            range_1 = [-0.1, 0.1]
            range_2 = [-0.1, 0.1]
        elif self.cfg.task_mode == "up_down":
            range_1 = [-0.1, 0.1]
            range_2 = [3.14 - 0.1, 3.14 + 0.1]
        elif self.cfg.task_mode == "down_up":
            range_1 = [3.14 - 0.1, 3.14 + 0.1]
            range_2 = [3.14 - 0.1, 3.14 + 0.1]
        
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            range_1[0], range_1[1], joint_pos[:, self._pole_dof_idx].shape, joint_pos.device
        )
        joint_pos[:, self._pendulum_dof_idx] += sample_uniform(
            range_2[0], range_2[1], joint_pos[:, self._pendulum_dof_idx].shape, joint_pos.device
        )
        
        # 【修改处】删掉了重复获取 joint_vel 和 default_root_state 的代码

        # 3. 加上环境的绝对坐标偏移量
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # 4. 写回本地缓存
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # 5. 写回物理引擎
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

@torch.jit.script
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_cart_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_pos: float,
    rew_scale_pole_vel: float,
    rew_scale_pendulum_pos: float,
    rew_scale_pendulum_vel: float,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    pendulum_pos: torch.Tensor,
    pendulum_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
    target_p1: float,
    target_p2: float,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    
    error_p1 = normalize_angle(pole_pos - target_p1)
    rew_pole_up = torch.exp(-torch.square(error_p1) / 0.25)
    error_p2 = normalize_angle(pole_pos + pendulum_pos - target_p2)
    rew_pendulum_up = torch.exp(-torch.square(error_p2) / 0.25)
    
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    rew_pendulum_vel = rew_scale_pendulum_vel * torch.sum(torch.abs(pendulum_vel).unsqueeze(dim=1), dim=-1)

    total_reward = rew_alive + rew_termination + (10.0 * rew_pole_up) + (10.0 * rew_pendulum_up) + rew_cart_vel + rew_pole_vel + rew_pendulum_vel
    
    return total_reward
