from envs.draftEnv import FirstN
import pandas as pd
from stable_baselines3 import DQN


def test_load_model(env, base_model, location="results/agents/dqn", iters=1000):
    env.mode = "test"
    env.episode = 0
    base_model.load(location)
    obs = env.reset()
    for i in range(iters):
        action, _states = base_model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if i % 100 == 0:
            env.render()
            env.score = []
        if done:
            obs = env.reset()