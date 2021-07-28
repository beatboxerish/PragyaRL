import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class FirstN(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, n_points=10, plot_results=None, norm=True):
        super(FirstN, self).__init__()
        self.df = df
        self.n_points = n_points
        self.norm = norm
        self.plot_results = plot_results
        self.total_strokes = df.stroke.max()
        self.indices = self.df.stroke.unique()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(n_points, 2), dtype=np.uint8)  # correct this

        self.curr_id = 0
        self.curr_df = None
        self.label = None
        self.curr_step = 0
        self.score = []
        self.mode = None
        self.episode = 0

    def reset(self):
        self.episode += 1
        shape_check = -1  # to make sure that size of df is more than the specified size
        while shape_check < self.n_points:
            self.curr_id = np.random.choice(len(self.indices), 1)[0]
            self.curr_id = self.indices[self.curr_id]
            self.curr_df = self.df[self.df.stroke == self.curr_id]
            shape_check = self.curr_df.shape[0]

        self.label = self.curr_df.direction_int.unique()[0]
        if self.norm:
            ss = StandardScaler()
            self.curr_df = ss.fit_transform(self.curr_df.iloc[:, 0:2]).astype(float)
        else:
            self.curr_df = self.curr_df.values[:, :2].astype(float)
        return self.next_observation()

    def next_observation(self):
        # obs = self.curr_df.iloc[:self.n_points, 0:2].values
        obs = self.curr_df[:self.n_points]
        return obs

    def step(self, action):
        reward = 0
        if self.label == action:
            reward += 1
        else:
            reward -= 1
        self.score.append(reward)

        done = True
        obs = None

        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(range(len(self.score)), self.score)
        if self.plot_results:
            plt.savefig(f"{self.plot_results}render_{self.mode}_{str(self.episode)}.png")
