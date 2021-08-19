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
# from pybullet_envs.gym_locomotion_envs import HopperBulletEnv
# from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv
# from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
# from pybullet_envs.gym_locomotion_envs import AntBulletEnv
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv



p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
#don't create a ground plane, to allow for gaps etc
p.resetSimulation()
#p.createCollisionShape(p.GEOM_PLANE)
#p.createMultiBody(0,0)
#p.resetDebugVisualizerCamera(5,75,-26,[0,0,1]);

# set a fixed camera
p.resetDebugVisualizerCamera(15, -346, -16, [-15, 0, 1])

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

sphereRadius = 0.05
colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)

#a few different ways to create a mesh:

#convex mesh from obj
stoneId = p.createCollisionShape(p.GEOM_MESH, fileName="stone.obj")

boxHalfLength = 0.7
boxHalfWidth = 2.5
boxHalfHeight = 0.1
segmentLength = 10
stoneId = p.createCollisionShape(p.GEOM_MESH, fileName="stone.obj")
colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])


# creating the obstacles
mass = 1
visualShapeId = -1

segmentStart = 0

for i in range(segmentLength):
  p.createMultiBody(baseMass=0,
                    baseCollisionShapeIndex=colBoxId,
                    basePosition=[segmentStart, 0, -0.1])
  segmentStart = segmentStart - 1

# if you set it to zero segment starts again
# segmentStart = 0
for i in range(segmentLength):
  height = 0
  if (i % 2):
    height = 1.0
  p.createMultiBody(baseMass=0,
                    baseCollisionShapeIndex=colBoxId,
                    basePosition=[segmentStart, 0, -0.1 + height])
  segmentStart = segmentStart - 1

for i in range(segmentLength):
    height = i
    colBoxId2 = p.createCollisionShape(p.GEOM_BOX,
                                      halfExtents=[2, boxHalfWidth, height])
    p.createMultiBody(baseMass=0,
                    baseCollisionShapeIndex=colBoxId2,
                    basePosition=[segmentStart, 0, -0.1 + height])
    segmentStart = segmentStart - 1

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

p.setGravity(0,0,-5)
cubeStartPos = [0,-1,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,-1,0])
humanoidId = p.loadURDF("humanoid.urdf",cubeStartPos, cubeStartOrientation)
cubeStartPos = [0,-1,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,1,0])
R2D2Id = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)

while (1):
    p.stepSimulation()
    time.sleep(1./240.)
    camData = p.getDebugVisualizerCamera()
    viewMat = camData[2]
    projMat = camData[3]
    p.getCameraImage(256,
                   256,
                   viewMatrix=viewMat,
                   projectionMatrix=projMat,
                   renderer=p.ER_BULLET_HARDWARE_OPENGL)
    keys = p.getKeyboardEvents()
    p.stepSimulation()
    #print(keys)
cubePos, cubeOrn = p.getBasePositionAndOrientation(humanoidId)
print(cubePos,cubeOrn)
p.disconnect()
