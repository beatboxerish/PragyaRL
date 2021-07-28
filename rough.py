from envs.draftEnv import FirstN
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


# # # remapping the dataset
# df = pd.read_csv("datasets/stroke_gesture_df.csv")
# df["direction_int"] = df["direction_int"].map({1:0, 2:1})
# print("Minimum Number of data points for any given stroke:", df.groupby("stroke").size().min())
# df.to_csv("datasets/stroke_gesture_df_remapped.csv", index=False)

# # # testing a loaded model
# df = pd.read_csv("datasets/stroke_gesture_df_remapped.csv", index_col=0)
# env = FirstN(df, 10)
# model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
#
# model.load("results/agents/dqn")
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     if i % 100 == 0:
#         env.render()
#         env.score = []
#     if done:
#         obs = env.reset()

