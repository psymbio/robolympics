import pybullet as p
import time
import math
import pybullet_data
# Imports
import os
import shutil
import matplotlib
import pybullet_envs

from acme.utils import loggers
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.agents.tf.d4pg import D4PG
from acme.agents.tf.ddpg import DDPG
from acme.agents.tf.dmpo import DistributionalMPO
from acme import wrappers, specs, environment_loop

import numpy as np
import sonnet as snt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from google.colab import drive
from IPython.display import HTML
import pybullet as p
import time
import math
import pybullet_data


from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
import numpy as np
import pybullet


from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv

def make_networks_d4pg(action_spec,
                       policy_layer_sizes=(256, 256, 256),
                       critic_layer_sizes=(512, 512, 256),
                       vmin=-150.,
                       vmax=150.,
                       num_atoms=51,
                      ):
  """Networks for D4PG agent."""
  action_size = np.prod(action_spec.shape, dtype=int)

  policy_network = snt.Sequential([
      tf2_utils.batch_concat,
      networks.LayerNormMLP(layer_sizes=policy_layer_sizes + (action_size,)),
      networks.TanhToSpec(spec=action_spec)
      ])
  critic_network = snt.Sequential([
      networks.CriticMultiplexer(
          action_network=networks.ClipToSpec(action_spec),
          critic_network=networks.LayerNormMLP(
              layer_sizes=critic_layer_sizes,
              activate_final=True),
      ),
      networks.DiscreteValuedHead(vmin=vmin,
                                  vmax=vmax,
                                  num_atoms=num_atoms)
      ])

  return policy_network, critic_network


def make_networks_ddpg(action_spec,
                       policy_layer_sizes=(256, 256, 256),
                       critic_layer_sizes=(512, 512, 256),
                      ):
  """Networks for DDPG agent."""
  action_size = np.prod(action_spec.shape, dtype=int)

  policy_network = snt.Sequential([
      tf2_utils.batch_concat,
      networks.LayerNormMLP(layer_sizes=policy_layer_sizes + (action_size,)),
      networks.TanhToSpec(spec=action_spec)
      ])
  critic_network = networks.CriticMultiplexer(
          action_network=networks.ClipToSpec(action_spec),
          critic_network=networks.LayerNormMLP(
              layer_sizes=critic_layer_sizes + (1,),
              activate_final=False),
              )

  return policy_network, critic_network


def make_networks_dmpo(action_spec,
                       policy_layer_sizes=(256, 256, 256),
                       critic_layer_sizes=(512, 512, 256),
                       vmin=-150.,
                       vmax=150.,
                       num_atoms=51,
                      ):
  """Networks for DMPO agent."""
  action_size = np.prod(action_spec.shape, dtype=int)

  policy_network = snt.Sequential([
      tf2_utils.batch_concat,
      networks.LayerNormMLP(layer_sizes=policy_layer_sizes,
                            activate_final=True),
      networks.MultivariateNormalDiagHead(
          action_size,
          min_scale=1e-6,
          tanh_mean=False,
          init_scale=0.7,
          fixed_scale=False,
          use_tfd_independent=True)
  ])

  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = networks.CriticMultiplexer(
      action_network=networks.ClipToSpec(action_spec),
      critic_network=networks.LayerNormMLP(layer_sizes=critic_layer_sizes,
                                           activate_final=True),
                                           )
  critic_network = snt.Sequential([
                                   critic_network,
                                   networks.DiscreteValuedHead(vmin=vmin,
                                                               vmax=vmax,
                                                               num_atoms=num_atoms)
                                   ])

  return policy_network, critic_network


class Humanoid(HumanoidBulletEnv):

  def __init__(self, render=False, episode_steps=1000):
    """Modifies `__init__` in `HopperBulletEnv` parent class."""
    self.episode_steps = episode_steps
    super().__init__(render=render)

  def reset(self):
    """Modifies `reset` in `WalkerBaseBulletEnv` base class."""
    self.step_counter = 0
    return super().reset()

  def _isDone(self):
    """Modifies `_isDone` in `WalkerBaseBulletEnv` base class."""
    return (self.step_counter == self.episode_steps
            or super()._isDone())

  def step(self, a):
    """Fully overrides `step` in `WalkerBaseBulletEnv` base class."""

    self.step_counter += 1

    # if multiplayer, action first applied to all robots,
    # then global step() called, then _step() for all robots
    # with the same actions
    if not self.scene.multiplayer:
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    # state[0] is body height above ground, body_rpy[1] is pitch
    self._alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z,
                                               self.robot.body_rpy[1]))
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)

    feet_collision_cost = 0.0
    for i, f in enumerate(self.robot.feet):
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      # print(self.ground_ids)
      if (self.ground_ids & contact_ids):
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0
    # print(self.robot.feet_contact)
    ###### MY OWN CODE (THITI) #######################################
    # print(self.robot.feet_contact)
    jump = 1.0 if np.any(self.robot.feet_contact > 0) else 0.0


    # jump = 0.0 if self.robot.feet_contact > 0 else 1.0

    # jump = 0.0
    how_far = np.sqrt(self.robot.walk_target_x**2 + self.robot.walk_target_y**2) \
              - self.robot.walk_target_dist
    electricity_cost = self.electricity_cost * float(
        np.abs(a * self.robot.joint_speeds).mean())
    electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

    joints_at_limit_cost = float(10 * self.joints_at_limit_cost * self.robot.joints_at_limit)

    self.rewards = [
                    self._alive, progress, electricity_cost,
                    joints_at_limit_cost, feet_collision_cost,
                    jump, how_far
                    ]
    self.HUD(state, a, done)
    self.reward += sum(self.rewards)
    return state, sum(self.rewards), bool(done), {}



p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
#don't create a ground plane, to allow for gaps etc
p.resetSimulation()
#p.createCollisionShape(p.GEOM_PLANE)
#p.createMultiBody(0,0)
#p.resetDebugVisualizerCamera(5,75,-26,[0,0,1]);

# set a fixed camera
# p.resetDebugVisualizerCamera(15, -346, -16, [-15, 0, 1])

p.loadSDF('stadium.sdf')
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# env = Humanoid(render=True)
# env = wrappers.GymWrapper(env)
# env = wrappers.SinglePrecisionWrapper(env)
#
# action_spec = env.action_spec()  # Specifies action shape and dimensions.
# env_spec = specs.make_environment_spec(env)  # Environment specifications.
# env.robot.walk_target_x, env.robot.walk_target_y = 300, 0
# print("heyyyy", env.robot.walk_target_x, env.robot.walk_target_y)

cubeStartPos = [0,-1,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,-1,0])
humanoidId = p.loadURDF("humanoid.urdf",cubeStartPos, cubeStartOrientation)
p.setGravity(0,0,-5)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

while (1):
    p.stepSimulation()
    time.sleep(1./240.)
    # camData = p.getDebugVisualizerCamera()
    # viewMat = camData[2]
    # projMat = camData[3]
    # p.getCameraImage(256,
    #                256,
    #                viewMatrix=viewMat,
    #                projectionMatrix=projMat,
    #                renderer=p.ER_BULLET_HARDWARE_OPENGL)
    keys = p.getKeyboardEvents()
    p.stepSimulation()
    #print(keys)
p.disconnect()
