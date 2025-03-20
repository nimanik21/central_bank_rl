
import gym
import numpy as np
import pandas as pd
import pickle
import os
from gym import spaces
from config import MODELS_DIR, PROCESSED_DATA_DIR  # Import path setup

class CentralBankEnv(gym.Env):
    def __init__(self, model_name="varmax_model.pkl", data_name="train_data_diff.csv", optimal_lag=10, episode_length=60):
        super().__init__()

        # Load the VARMAX model
        model_path = os.path.join(MODELS_DIR, model_name)
        with open(model_path, "rb") as f:
            self.varmax_model = pickle.load(f)

        # Load the training dataset
        train_data_path = os.path.join(PROCESSED_DATA_DIR, data_name)
        self.train_data = pd.read_csv(train_data_path, index_col="date", parse_dates=True)

        # Define exogenous and endogenous variables
        self.exogenous_vars = ["FEDFUNDS"]
        self.endogenous_vars = [col for col in self.train_data.columns if col not in self.exogenous_vars]

        if len(self.endogenous_vars) < 1:
            raise ValueError("Dataset must contain at least one endogenous variable.")

        self.optimal_lag = optimal_lag
        self.episode_length = episode_length

        num_features = len(self.endogenous_vars) + len(self.exogenous_vars)
        self.expected_shape = (self.optimal_lag, num_features)

        # Define action and observation spaces
        self.action_mapping = np.array([-0.50, -0.25, 0, 0.25, 0.50])
        self.action_space = spaces.Discrete(len(self.action_mapping))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.expected_shape, dtype=np.float32)

        # Define FFR limits
        self.ffr_min, self.ffr_max = self.train_data["FEDFUNDS"].min(), self.train_data["FEDFUNDS"].max()

        # Find CPI index
        if "CPIAUCSL" in self.endogenous_vars:
            self.inflation_idx = self.endogenous_vars.index("CPIAUCSL")
        else:
            raise KeyError("CPIAUCSL not found in dataset.")

        self.steps = 0
        self.episode_rewards = []
        self.current_episode_rewards = 0

    def reset(self):
        if self.steps > 0:
            self.episode_rewards.append(self.current_episode_rewards)

        self.steps = 0
        self.current_episode_rewards = 0

        max_start_idx = len(self.train_data) - self.optimal_lag - 1
        if max_start_idx <= 0:
            raise ValueError(f"Not enough training data. Dataset size: {len(self.train_data)}, required minimum: {self.optimal_lag + 1}")

        start_idx = np.random.randint(0, max_start_idx)
        self.history = self.train_data.iloc[start_idx : start_idx + self.optimal_lag][self.endogenous_vars + self.exogenous_vars].to_numpy(copy=False)
        self.last_ffr = self.history[-1, -1]

        return self.history

    def step(self, action):
        ffr_change = self.action_mapping[action]
        new_ffr = np.clip(self.last_ffr + ffr_change, self.ffr_min, self.ffr_max)

        next_state_df = self.varmax_model.forecast(steps=1, exog=np.array([[new_ffr]]))
        next_state = next_state_df[self.endogenous_vars].iloc[0].values.reshape(1, -1)

        new_row = np.hstack([next_state, np.array([[new_ffr]])])
        self.history = np.vstack([self.history[1:], new_row])
        self.last_ffr = new_ffr

        if self.history.shape != self.expected_shape:
            raise ValueError(f"History shape mismatch: Expected {self.expected_shape}, got {self.history.shape}")

        reward = self._reward(next_state.flatten(), action)
        self.current_episode_rewards += reward

        self.steps += 1
        done = self.steps >= self.episode_length

        return self.history, reward, done, {}

    def _reward(self, next_state, action):
        inflation = next_state[self.inflation_idx]
        pi_target = 0.02
        alpha, lamb = 0.5, 0.1
        return - (alpha * (inflation - pi_target)**2 + lamb * (action**2))

    def get_episode_rewards(self):
        return self.episode_rewards
