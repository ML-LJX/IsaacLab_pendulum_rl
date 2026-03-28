# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
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

# 【修复报错的核心区域】：不从外部导入不存在的配置，而是借用双摆的配置
from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG

# 获取当前 Python 文件所在的绝对路径
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 移花接木：复制双摆的配置，但把模型路径指向你同目录下的三摆 USD 文件
_TRIPLE_PENDULUM_CFG = CART_DOUBLE_PENDULUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
_TRIPLE_PENDULUM_CFG.spawn.usd_path = f"{_CURRENT_DIR}/cart_triple_pendulum.usd"


@configclass
class CartTriplePendulumEnvCfg(DirectRLEnvCfg):
    
    # 默认设定为三杆全上的终极挑战
    task_mode = "up_up_up" 
    target_state = [0.0, 0.0, 0.0, 0.0] 
    
    # env
    decimation = 2
    episode_length_s = 5.0
    
    # action_space 改为 1。只施加力矩给小车(Cart)
    action_space = 1
    # observation_space 改为 10
    observation_space = 10
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    # 【应用修复后的配置】
    robot_cfg: ArticulationCfg = _TRIPLE_PENDULUM_CFG
    
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    pendulum_dof_name = "pole_to_pendulum"
    # 绑定 USD 中第三根杆子的关节名
    pendulum3_dof_name = "pole_to_pendulum_3"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # reset
    max_cart_pos = 3.0  
    initial_pole_angle_range = [-0.25, 0.25]  
    initial_pendulum_angle_range = [-0.25, 0.25]  
    initial_pendulum3_angle_range = [-0.25, 0.25]

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
    rew_scale_pendulum3_pos = -1.0
    rew_scale_pendulum3_vel = -0.01


class CartTriplePendulumEnv(DirectRLEnv):
    cfg: CartTriplePendulumEnvCfg

    def __init__(self, cfg: CartTriplePendulumEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        self._pendulum_dof_idx, _ = self.robot.find_joints(self.cfg.pendulum_dof_name)
        self._pendulum3_dof_idx, _ = self.robot.find_joints(self.cfg.pendulum3_dof_name)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
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
        pendulum3_joint_pos = normalize_angle(self.joint_pos[:, self._pendulum3_dof_idx[0]].unsqueeze(dim=1))
        
        obs = torch.cat(
            (
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                pole_joint_pos,
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                normalize_angle(pole_joint_pos + pendulum_joint_pos), 
                pendulum_joint_pos,                                   
                self.joint_vel[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1),
                normalize_angle(pole_joint_pos + pendulum_joint_pos + pendulum3_joint_pos), 
                pendulum3_joint_pos,                                                        
                self.joint_vel[:, self._pendulum3_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.task_mode == "up_up_up":
            target_p1, target_p2, target_p3 = 0.0, 0.0, 0.0
        else:
            target_p1, target_p2, target_p3 = 0.0, 0.0, 0.0
            
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_cart_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_pole_vel,
            self.cfg.rew_scale_pendulum_pos,
            self.cfg.rew_scale_pendulum_vel,
            self.cfg.rew_scale_pendulum3_pos, 
            self.cfg.rew_scale_pendulum3_vel, 
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            normalize_angle(self.joint_pos[:, self._pole_dof_idx[0]]),
            self.joint_vel[:, self._pole_dof_idx[0]],
            normalize_angle(self.joint_pos[:, self._pendulum_dof_idx[0]]),
            self.joint_vel[:, self._pendulum_dof_idx[0]],
            normalize_angle(self.joint_pos[:, self._pendulum3_dof_idx[0]]), 
            self.joint_vel[:, self._pendulum3_dof_idx[0]],                  
            self.reset_terminated,
            target_p1, 
            target_p2,
            target_p3, 
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        
        target_p1 = 0.0
        error_p1 = normalize_angle(self.joint_pos[:, self._pole_dof_idx] - target_p1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(error_p1) > math.pi / 2, dim=1)   
            
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        
        if self.cfg.task_mode == "up_up_up":
            range_1 = [-0.1, 0.1]
            range_2 = [-0.1, 0.1]
            range_3 = [-0.1, 0.1]
        else:
            range_1 = [-0.1, 0.1]
            range_2 = [-0.1, 0.1]
            range_3 = [-0.1, 0.1]
        
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            range_1[0], range_1[1], joint_pos[:, self._pole_dof_idx].shape, joint_pos.device
        )
        joint_pos[:, self._pendulum_dof_idx] += sample_uniform(
            range_2[0], range_2[1], joint_pos[:, self._pendulum_dof_idx].shape, joint_pos.device
        )
        joint_pos[:, self._pendulum3_dof_idx] += sample_uniform(
            range_3[0], range_3[1], joint_pos[:, self._pendulum3_dof_idx].shape, joint_pos.device
        )

        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

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
    rew_scale_pendulum3_pos: float, 
    rew_scale_pendulum3_vel: float, 
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    pendulum_pos: torch.Tensor,
    pendulum_vel: torch.Tensor,
    pendulum3_pos: torch.Tensor,    
    pendulum3_vel: torch.Tensor,    
    reset_terminated: torch.Tensor,
    target_p1: float,
    target_p2: float,
    target_p3: float,               
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    
    error_p1 = normalize_angle(pole_pos - target_p1)
    rew_pole_up = torch.exp(-torch.square(error_p1) / 0.25)
    
    error_p2 = normalize_angle(pole_pos + pendulum_pos - target_p2)
    rew_pendulum_up = torch.exp(-torch.square(error_p2) / 0.25)
    
    error_p3 = normalize_angle(pole_pos + pendulum_pos + pendulum3_pos - target_p3)
    rew_pendulum3_up = torch.exp(-torch.square(error_p3) / 0.25)
    
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    rew_pendulum_vel = rew_scale_pendulum_vel * torch.sum(torch.abs(pendulum_vel).unsqueeze(dim=1), dim=-1)
    rew_pendulum3_vel = rew_scale_pendulum3_vel * torch.sum(torch.abs(pendulum3_vel).unsqueeze(dim=1), dim=-1)

    total_reward = (
        rew_alive + rew_termination 
        + (15.0 * rew_pole_up) 
        + (10.0 * rew_pendulum_up) 
        + (5.0 * rew_pendulum3_up) 
        + rew_cart_vel + rew_pole_vel + rew_pendulum_vel + rew_pendulum3_vel
    )
    
    return total_reward