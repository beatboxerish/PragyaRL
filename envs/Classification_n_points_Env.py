import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def train_test(data_idx):
    train_idx, test_idx = train_test_split(data_idx, test_size=0.2, random_state=0, shuffle=True)
    return train_idx, test_idx


class FirstN(gym.Env):
    """Selects the firstN points of a gesture that will be shown to the algorithm. These will be used to classify
    the gesture.
    The actions are 0 and 1 which stand for different gestures. Two columns are provided to the algorithm: X and Y
    coordinates.
    Make norm=True to normalize the data."""
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
        self.train, self.test = train_test(self.indices)
        self.do_train = True
        self.do_test = False
        assert self.do_train != self.do_test

        self.curr_id = 0
        self.curr_df = None
        self.label = None
        self.curr_step = 0
        self.score = []
        self.mode = None
        self.episode = 0

        self.df_describe()

    def df_describe(self):
        df = self.df
        print(" || DATASET DESCRIPTION ||")
        print("Shape:", df.shape)
        print("Columns:", df.columns)
        print("Unique Gestures:", df.direction.unique())
        print("Number of Individual Gestures:", df.stroke.nunique())
        print("Train and Test Shape:", self.train.shape, self.test.shape)
        print("\n")

    def reset(self):
        assert self.do_train != self.do_test
        self.episode += 1
        if self.do_train:
            choices = self.train
        else:
            choices = self.test
        shape_check = -1  # to make sure that size of df is more than the specified size
        while shape_check < self.n_points:
            self.curr_id = np.random.choice(choices)
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
