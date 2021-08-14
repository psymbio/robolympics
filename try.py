from absl import app

import acme
from acme import wrappers
from acme import specs

from acme.agents.jax import dqn

import haiku as hk
import jax.numpy as jnp

import gym
import nle

def main(_):
    raw_env = gym.make("NetHackScore-v0")
    env = wrappers.GymWrapper(raw_env)
    env = wrappers.SinglePrecisionWrapper(env)

    env_spec = specs.make_environment_spec(env)

    def module(x):
        x = jnp.float32(x['glyphs'])
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.nets.MLP([50, 50, 23])
        ])
        return mlp(x)

    network = hk.without_apply_rng(hk.transform(module, apply_rng=True))
    agent = dqn.DQN(environment_spec=env_spec, network=network)

    loop = acme.EnvironmentLoop(env, agent)
    loop.run(num_episodes=10)

if __name__ == "__main__":
    app.run(main)
