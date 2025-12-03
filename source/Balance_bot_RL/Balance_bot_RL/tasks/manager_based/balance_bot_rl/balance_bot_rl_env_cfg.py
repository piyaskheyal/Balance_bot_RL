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
from isaaclab.managers import CommandTermCfg as CmdTerm
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
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.85,  # Grip when starting from stop
                dynamic_friction=0.80, # Grip when already moving
                friction_combine_mode="multiply", # Intelligent combining
                restitution=0.0,      # 0.0 means "no bouncing"
            ),
        ),
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
class CommandsCfg:
    """Command specifications for the RL agent."""

    # Instead of defining a generic CmdTerm with func=..., 
    # we use the specific Config class provided by Isaac Lab.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.0, 5.0),
        debug_vis=True,
        
        # The ranges are defined inside a nested "Ranges" class or dict
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), 
            lin_vel_y=(0.0, 0.0),   # ZERO for two-wheeled robots
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )


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
        
        # 5. Target Velocity Command
        velocity_command = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
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
        weight=1.0 # Increased slightly to encourage long episodes
    )

    # (2) Upright Penalty
    # We relax this slightly so the robot feels free to lean into the run.
    upright_penalty = RewTerm(
        func=mdp.root_tilt_l2, 
        weight=-2.0, 
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # (3) Velocity Tracking
    # Logic: If I command 1.0 m/s and you go 1.0 m/s, penalty is 0.
    velocity_tracking = RewTerm(
        func=mdp.track_lin_vel_xy_l2,
        weight=-1.0, # Strong incentive to match speed
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # (4) Turn Tracking (Steering) <--- NEW
    # Logic: If I command "Turn Left", you must turn left.
    turn_tracking = RewTerm(
        func=mdp.track_ang_vel_z_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # (5) Smoothness (Anti-Jitter)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05, # Tuned down slightly to allow quick reactions
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
    commands: CommandsCfg = CommandsCfg()

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