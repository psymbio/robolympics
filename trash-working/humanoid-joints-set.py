import pybullet as p
import time
import math
import pybullet_data
import json
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
p.setGravity(0,0,-5)

timeStep = 1. / 600.
p.setPhysicsEngineParameter(fixedTimeStep=timeStep)
path = "data/motions/humanoid3d_walk.txt"
print("path	=	", path)
with open(path, 'r') as f:
  motion_dict = json.load(f)
print("len motion=", len(motion_dict))
print(motion_dict['Loop'])
numFrames = len(motion_dict['Frames'])
print("#frames = ", numFrames)
frameId = p.addUserDebugParameter("frame", 0, numFrames - 1, 0)

erpId = p.addUserDebugParameter("erp", 0, 1, 0.2)

kpMotorId = p.addUserDebugParameter("kpMotor", 0, 1, .2)
forceMotorId = p.addUserDebugParameter("forceMotor", 0, 2000, 1000)

jointTypes = [
    "JOINT_REVOLUTE", "JOINT_PRISMATIC", "JOINT_SPHERICAL", "JOINT_PLANAR", "JOINT_FIXED"
]

startLocations = [[28,-20,0]]
# started of running field
cubeStartPos = [28,-20,5]
cubeStartOrientation = p.getQuaternionFromEuler([0,-1,0])
humanoid = p.loadURDF("humanoid.urdf",cubeStartPos, cubeStartOrientation)

#humanoid_fix = p.createConstraint(humanoid, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], startLocations[0], [0, 0, 0, 1])

startPose = [
    2, 0.847532, 0, 0.9986781045, 0.01410400148, -0.0006980000731, -0.04942300517, 0.9988133229,
    0.009485003066, -0.04756001538, -0.004475001447, 1, 0, 0, 0, 0.9649395871, 0.02436898957,
    -0.05755497537, 0.2549218909, -0.249116, 0.9993661511, 0.009952001505, 0.03265400494,
    0.01009800153, 0.9854981188, -0.06440700776, 0.09324301124, -0.1262970152, 0.170571,
    0.9927545808, -0.02090099117, 0.08882396249, -0.07817796699, -0.391532, 0.9828788495,
    0.1013909845, -0.05515999155, 0.143618978, 0.9659421276, 0.1884590249, -0.1422460188,
    0.105854014, 0.581348
]

startVel = [
    1.235314324, -0.008525509087, 0.1515293946, -1.161516553, 0.1866449799, -0.1050802848, 0,
    0.935706195, 0.08277326387, 0.3002461862, 0, 0, 0, 0, 0, 1.114409628, 0.3618553952,
    -0.4505575061, 0, -1.725374735, -0.5052852598, -0.8555179722, -0.2221173515, 0, -0.1837617357,
    0.00171895706, 0.03912837591, 0, 0.147945294, 1.837653345, 0.1534535548, 1.491385941, 0,
    -4.632454387, -0.9111172777, -1.300648184, -1.345694622, 0, -1.084238535, 0.1313680236,
    -0.7236998534, 0, -0.5278312973
]

p.resetBasePositionAndOrientation(humanoid, startLocations[0], [0, 0, 0, 1])

jointIds = []
paramIds = []

p.setRealTimeSimulation(1)
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
    #print(keys)
p.disconnect()
