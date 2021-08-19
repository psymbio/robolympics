import pybullet as p
import time
import math
import pybullet_data
# Imports
import os
import shutil
import matplotlib
import pybullet_envs

# from acme.utils import loggers
# from acme.tf import networks
# from acme.tf import utils as tf2_utils
# from acme.agents.tf.d4pg import D4PG
# from acme.agents.tf.ddpg import DDPG
# from acme.agents.tf.dmpo import DistributionalMPO
# from acme import wrappers, specs, environment_loop

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

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
#don't create a ground plane, to allow for gaps etc
p.resetSimulation()
#p.createCollisionShape(p.GEOM_PLANE)
#p.createMultiBody(0,0)
p.resetDebugVisualizerCamera(5,75,-26,[0,0,1]);

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
p.setGravity(0,0,-5)
# started of running field
cubeStartPos = cubeStartPos = [28,-20,5]
cubeStartOrientation = p.getQuaternionFromEuler([0,-1,0])
humanoid = p.loadURDF("humanoid.urdf",cubeStartPos, cubeStartOrientation)

gravId = p.addUserDebugParameter("gravity", -10, 10, -10)
jointIds = []
paramIds = []

p.setPhysicsEngineParameter(numSolverIterations=10)
p.changeDynamics(humanoid, -1, linearDamping=0, angularDamping=0)

for j in range(p.getNumJoints(humanoid)):
  p.changeDynamics(humanoid, j, linearDamping=0, angularDamping=0)
  info = p.getJointInfo(humanoid, j)
  #print(info)
  jointName = info[1]
  jointType = info[2]
  if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
    jointIds.append(j)
    paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))

p.setRealTimeSimulation(1)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

while (1):
    p.stepSimulation()
    time.sleep(1./240.)
    p.setGravity(0, 0, p.readUserDebugParameter(gravId))
    # camData = p.getDebugVisualizerCamera()
    # viewMat = camData[2]
    # projMat = camData[3]
    # p.getCameraImage(256,
    #                256,
    #                viewMatrix=viewMat,
    #                projectionMatrix=projMat,
    #                renderer=p.ER_BULLET_HARDWARE_OPENGL)
    for i in range(len(paramIds)):
	    c = paramIds[i]
	    targetPos = p.readUserDebugParameter(c)
	    p.setJointMotorControl2(humanoid, jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)
	    keys = p.getKeyboardEvents()
	    p.stepSimulation()
	    #print(keys)
p.disconnect()
