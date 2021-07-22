from envs.draftEnv import FirstN
import pandas as pd
from utils.test import test_load_model
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


df = pd.read_csv("datasets/stroke_gesture_df_remapped.csv", index_col=0)
env = FirstN(df, 20, plot_results="results/plots/")
model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)

env.mode = "train"
model.learn(total_timesteps=1000)
model.save("results/agents/dqn")
# # Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# print(mean_reward)

test_load_model(env, model, "results/agents/dqn", 1000)
