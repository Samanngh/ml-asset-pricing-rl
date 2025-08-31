import random
import numpy as np
import pandas as pd


class Env:
    """
    Minimal portfolio environment operating on DAILY SIMPLE RETURNS.
    Expects:
      data/datasets/train.csv
      data/datasets/test.csv

    The CSVs should have Date as index and one column per asset (daily returns).
    """

    def __init__(
        self,
        data_path: str = "data/datasets",
        context: str = "train",
        yearly_return: float = 0.05,
        T: int = 252,
        lookback: int = 30,
    ):
        if context == "train":
            self.returns = pd.read_csv(f"{data_path}/train.csv", index_col=0, parse_dates=True)
        elif context == "test":
            self.returns = pd.read_csv(f"{data_path}/test.csv", index_col=0, parse_dates=True)
        else:
            raise ValueError("context must be 'train' or 'test'")

        # Drop rows where ALL assets are zero (full-market holiday)
        self.returns = self.returns[~(self.returns == 0).all(axis=1)]

        self.assets_col = list(self.returns.columns)
        self.n_assets = len(self.assets_col)
        self.n_actions = self.n_assets

        self.T = T
        self.lookback = lookback
        self.history_len = self.returns.shape[0]

        self.yearly_return = yearly_return
        self.target_return = yearly_return * (self.T / 252)

        # trackers set in reset()
        self.t = 0
        self.state = None
        self.dates = None
        self.assets = None
        self.IV = 1.0
        self.PV = 1.0
        self.PV_balanced = 1.0
        self.AV = np.ones(self.n_assets)

        self.reset()

    def reset(self):
        """Reset environment to the start of a new episode."""
        max_start = self.returns.shape[0] - self.T - self.lookback - 1
        self.start = 0 if max_start <= 0 else int(random.uniform(0, max_start))
        self.indices = np.arange(self.start, self.start + self.T + self.lookback)

        self.t = 0
        self.prev_action = np.zeros(self.n_assets)
        self.IV = 1.0
        self.PV = 1.0
        self.PV_balanced = 1.0
        self.AV = np.ones(self.n_assets)
        self.dates = self.returns.index[self.indices]

        # Assets vector for the first action (just after the lookback window)
        self.assets = self.returns.loc[self.dates[self.lookback], self.assets_col].values

        # Logs
        self.portfolio_returns = []
        self.index_returns = []

        # State = [prev_weights ; past_returns_matrix] flattened
        prev_weights = (np.ones(self.n_assets) / self.n_assets).reshape((-1, 1))
        past_returns = self.returns.loc[self.dates[:self.lookback], self.assets_col].values.T
        state = np.concatenate((prev_weights, past_returns), axis=1)
        self.state = state.flatten()

        self.n_states = self.state.shape[0]
        return self.state

    def _to_numpy(self, action):
        """Accept numpy array or torch tensor for actions; return np.ndarray."""
        try:
            return action.detach().cpu().numpy()
        except AttributeError:
            return np.asarray(action)

    def step_assets(self):
        """Advance to the next day's returns aligned with the lookback window."""
        # Use the t-th step after the lookback window
        assets = self.returns.loc[self.dates[self.lookback + self.t], self.assets_col].values
        self.AV = self.AV * (1.0 + assets)
        return assets

    def step_state(self, action):
        """Build next state from action (as prev weights) and latest lookback returns."""
        a = self._to_numpy(action)
        prev_weights = a.reshape((-1, 1))

        past_returns = self.returns.loc[
            self.dates[self.t : (self.t + self.lookback)], self.assets_col
        ].values.T

        state = np.concatenate((prev_weights, past_returns), axis=1)
        self.state = state.flatten()
        return self.state

    def get_reward(self):
        """Reward = log(1 + portfolio_return)."""
        reward = np.log(1.0 + self.portfolio_return)
        done = bool(self.t == self.T)
        return reward, done

    def step(self, action):
        """Full transition given an action (weights)."""
        # Get today's returns vector (aligned)
        self.assets = self.step_assets()
        self.t += 1

        a = self._to_numpy(action)
        self.portfolio_return = float(np.dot(a, self.assets))
        self.portfolio_returns.append(self.portfolio_return)

        # Update portfolio values (compound correctly)
        self.PV = self.PV * (1.0 + self.portfolio_return)
        self.PV_balanced = self.PV_balanced * (1.0 + float(np.mean(self.assets)))

        # Build next state
        self.state = self.step_state(action)

        # Compute reward & done
        reward, done = self.get_reward()
        return self.state, reward, done
