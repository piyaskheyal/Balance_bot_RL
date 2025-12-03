# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_tilt_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize the deviation of the robot's Z-axis from the world Z-axis.
    
    Logic:
    1. Isaac Lab calculates 'projected_gravity_b'. This is the gravity vector 
       as seen by the robot. 
    2. If the robot is perfectly upright, gravity pulls straight down (-Z). 
       The vector is [0, 0, -1].
    3. If the robot tilts, the X and Y components of this vector become non-zero.
    4. We penalize the squared magnitude of X and Y (x^2 + y^2).
    """
    # Access the robot from the scene
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the gravity vector projected into the robot's body frame
    proj_grav = asset.data.projected_gravity_b
    
    # We take the X and Y components (index 0 and 1)
    xy_tilt = proj_grav[:, :2]
    
    # Calculate L2 norm (squared sum) of X and Y
    return torch.sum(torch.square(xy_tilt), dim=1)

def bad_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, limit_angle_deg: float) -> torch.Tensor:
    """
    Terminate if the robot tilts more than 'limit_angle_deg' from upright.
    """
    # 1. Get the robot
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 2. Get projected gravity (Z component only)
    proj_grav_z = asset.data.projected_gravity_b[:, 2]
    
    # 3. Calculate threshold
    limit_rad = torch.tensor(limit_angle_deg, device=env.device) * torch.pi / 180.0
    
    # The threshold is negative cosine of the angle.
    threshold = -torch.cos(limit_rad)
    
    # 4. Check if we violated the threshold
    return proj_grav_z > threshold

# 1. Linear Velocity Penalty (XY only)
def base_lin_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize movement in the X and Y directions (World Frame).
    We ignore Z (index 2) because we don't want to punish the robot for moving up/down 
    (which happens naturally as it balances).
    """
    # Get the robot
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get linear velocity of the root in world frame
    root_vel = asset.data.root_lin_vel_w
    
    # Slice to keep only X and Y (indices 0 and 1)
    vel_xy = root_vel[:, :2]
    
    # Calculate Squared L2 norm: x^2 + y^2
    return torch.sum(torch.square(vel_xy), dim=1)


# 2. Action Rate Penalty
def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Penalize the CHANGE in action between steps.
    This is what fixes the 'violent shaking'.
    """
    # The ActionManager automatically stores the previous step's action
    curr_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    
    # Calculate difference
    diff = curr_action - prev_action
    
    # Return squared difference
    return torch.sum(torch.square(diff), dim=1)


# 3. Joint Velocity Penalty
def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize high speeds on the wheels.
    Helps prevent the robot from accelerating infinitely.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get joint velocities
    # Shape: (num_envs, num_joints)
    j_vel = asset.data.joint_vel
    
    # Return squared sum of all joint velocities
    return torch.sum(torch.square(j_vel), dim=1)