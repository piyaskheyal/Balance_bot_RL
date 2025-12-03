# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from .balance_bot import BALANCE_BOT_CFG


##
# Scene definition
##


@configclass
class BalanceBotRlSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.06)),
    )

    # robot
    robot: ArticulationCfg = BALANCE_BOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", 
        joint_names=["Left_wheel_joint", "Right_wheel_joint"], 
        scale=10.0,
        )
    
@configclass
class ObservationsCfg:
    """Observation specifications for the RL agent."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy network."""

        # 1. Robot Tilt (Projected Gravity)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # 2. Joint Positions
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # 3. Joint Velocity
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # 4. Base Angular Velocity
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    
@configclass
class RewardsCfg:
    """Reward terms for the Balance Bot."""

    # (1) Constant running reward
    alive = RewTerm(
        func=mdp.is_alive, 
        weight=0.1
    )

    # (2) Primary task: Keep body upright (Pitch/Roll = 0)
    upright_penalty = RewTerm(
        func=mdp.root_tilt_l2, # This is the custom function from above
        weight=-5.0,       # High penalty to prioritize balance over everything else
        params={
            "asset_cfg": SceneEntityCfg("robot") # "robot" must match the name in your SceneCfg
        },
    )

    # (3) Penalize "Running Away"
    lin_vel_xy = RewTerm(
        func=mdp.base_lin_vel_xy_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # (4) Penalize "Jitter" (The Violent Shaking)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.1, # Critical for stopping vibration
    )

    # (5) Penalize High Wheel Speed
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Tilt Limit (70 degrees)
    bad_tilt = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "limit_angle_deg": 70.0  # The angle you requested
        },
    )


##
# Environment configuration
##


@configclass
class BalanceBotRlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BalanceBotRlSceneCfg = BalanceBotRlSceneCfg(num_envs=4096, env_spacing=1.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 15
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 1.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation