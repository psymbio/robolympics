from acme import environment_loop
from acme.tf import networks
from acme.adders import reverb as adders
from acme.agents.tf import actors as actors
from acme.datasets import reverb as datasets
from acme.wrappers import gym_wrapper
from acme import specs
from acme import wrappers
from acme.agents.tf import d4pg
from acme.agents import agent
from acme.tf import utils as tf2_utils
from acme.utils import loggers

import gym
import dm_env
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf

# Import dm_control if it exists.
try:
    from dm_control import suite
except (OSError, ModuleNotFoundError):
    pass

def main():
    environment, environment_spec = load_environment()

    agent = create_agent(environment_spec)

    run_training_loop(environment, agent)

def load_environment():
    environment = gym_wrapper.GymWrapper(gym.make('MountainCarContinuous-v0'))
    environment = wrappers.SinglePrecisionWrapper(environment)

    def render(env):
        return env.environment.render(mode='rgb_array')

    # Grab the spec of the environment.
    environment_spec = specs.make_environment_spec(environment)
    return environment, environment_spec

def create_agent(environment_spec):
    #@title Build agent networks

    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

    # Create the shared observation network; here simply a state-less operation.
    observation_network = tf2_utils.batch_concat

    # Create the deterministic policy network.
    policy_network = snt.Sequential([
        networks.LayerNormMLP((256, 256, 256), activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(environment_spec.actions),
    ])

    # Create the distributional critic network.
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP((512, 512, 256), activate_final=True),
        networks.DiscreteValuedHead(vmin=-150., vmax=150., num_atoms=51),
    ])

    # 3. Create a logger for agent specific diagnostics.
    agent_logger = loggers.TerminalLogger(label='agent', time_delta=10)

    # Create the D4PG agent.
    agent = d4pg.D4PG(
        environment_spec=environment_spec,
        policy_network=policy_network,
        critic_network=critic_network,
        observation_network=observation_network,
        logger=agent_logger,
        checkpoint=False
    )
    return agent

def run_training_loop(environment, agent):
    # Create a logger for agent specific diagnostics.
    env_loop_logger = loggers.TerminalLogger(label='env_loop', time_delta=10)

    env_loop = environment_loop.EnvironmentLoop(environment, agent, logger=env_loop_logger)
    env_loop.run(num_episodes=100)
