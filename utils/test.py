from envs.Classification_n_points_Env import FirstN
import pandas as pd
from stable_baselines3 import DQN
import matplotlib.pyplot as plt


def test_load_model(env, base_model, location="results/agents/dqn", iters=1000):
    env.mode = "test"
    env.episode = 0
    base_model.load(location)
    obs = env.reset()
    actions = []
    for i in range(iters):
        action, _states = base_model.predict(obs)
        actions.append(action)
        obs, rewards, done, info = env.step(action)
        if i % 100 == 0:
            # fig, ax = plt.subplots(1,1)
            # plt.plot(actions)
            env.render()
            env.score = []
            actions = []
        if done:
            obs = env.reset()