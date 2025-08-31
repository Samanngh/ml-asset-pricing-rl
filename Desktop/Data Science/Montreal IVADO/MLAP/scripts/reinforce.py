import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from scripts.env import Env
from scripts.policy import Policy


class REINFORCE:
    """
    Deterministic policy-gradient-style training:
      • Policy outputs weights via softmax (simplex).
      • Reward per step: log(1 + w · r_t).
      • Backprop through reward by computing it in torch BEFORE env.step().
    """

    def __init__(
        self,
        n_neurons: List[int],
        T: int,
        learning_rate: float = 1e-3,
        data_path: str = "data/datasets",
        mode: str = "training",
        model_path: str = None,
        lookback: int = 30,
        device: str = "cpu",
    ):
        # Environments
        self.env_train = Env(data_path=data_path, context="train", T=T, lookback=lookback)
        self.env_test = Env(data_path=data_path, context="test", T=T, lookback=lookback)

        # Policy
        n_inputs = self.env_train.n_states
        n_actions = self.env_train.n_actions if mode == "training" else self.env_test.n_actions
        self.policy = Policy(n_inputs, n_neurons, n_actions).to(device)
        if mode == "predicting" and model_path:
            self.policy.load(model_path)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.device = device
        self.T = T

    # ---------- checkpoint helpers ----------
    def save_checkpoint(self, path="models/checkpoint.pt", it=0, best=0.0):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "it": it,
                "model_state": self.policy.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "best_test": best,
            },
            path,
        )

    def load_checkpoint(self, path="models/checkpoint.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        return ckpt.get("it", 0), ckpt.get("best_test", 0.0)

    # ---------- action & returns helpers ----------
    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Deterministic action = policy(state)."""
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        w = self.policy(s).squeeze(0)  # (n_actions,)
        return w

    def _discounted_returns(self, rewards: List[torch.Tensor], gamma: float) -> torch.Tensor:
        """Compute normalized discounted returns G_t."""
        G = []
        running = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for r in reversed(rewards):
            running = r + gamma * running
            G.append(running)
        G = torch.stack(list(reversed(G)))  # shape (<=T,)
        if G.std() > 0:
            G = (G - G.mean()) / (G.std() + 1e-8)
        return G

    # ---------- episode generation ----------
    def generate_episode(self, env_name: str) -> Tuple[list, list, list, list, list]:
        """Play one episode in the given environment ('train' or 'test')."""
        assert env_name in ("train", "test")
        env = self.env_train if env_name == "train" else self.env_test

        states, actions, rewards = [], [], []
        PV, PV_balanced = [], []

        state = env.reset()

        for _ in range(self.T):
            # 1) action
            w = self.select_action(state).to(self.device).float()

            # 2) reward = log(1 + w·r_t)
            assets = torch.tensor(env.assets, dtype=torch.float32, device=self.device)
            portfolio = torch.dot(assets, w)
            reward = torch.log1p(portfolio)

            # 3) transition env
            new_state, _, done = env.step(w)

            # 4) log
            states.append(state)
            actions.append(w.detach().cpu().numpy())
            rewards.append(reward)

            state = new_state
            PV.append(env.PV)
            PV_balanced.append(env.PV_balanced)

            if done:
                break

        return states, actions, rewards, PV, PV_balanced

    # ---------- training & evaluation ----------
    def get_policy_loss(self, rewards: List[torch.Tensor], gamma: float) -> torch.Tensor:
        """Monte Carlo objective: maximize sum_t G_t → minimize -mean(G_t)."""
        G = self._discounted_returns(rewards, gamma)
        return -G.mean()

    def update_policy(self, policy_loss: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        return float(policy_loss.detach().cpu().numpy())

    def evaluate_policy(self, n_test_episodes: int = 5):
        """Average PV on train/test and test balanced benchmark."""
        avg_PV_train, avg_PV_test, avg_PV_bal = 0.0, 0.0, 0.0
        with torch.no_grad():
            for _ in range(n_test_episodes):
                self.generate_episode(env_name="train")
                self.generate_episode(env_name="test")
                avg_PV_train += self.env_train.PV / n_test_episodes
                avg_PV_test += self.env_test.PV / n_test_episodes
                avg_PV_bal += self.env_test.PV_balanced / n_test_episodes
        return avg_PV_train, avg_PV_test, avg_PV_bal

    def train_agent(
        self,
        gamma: float = 0.99,
        n_train_iterations: int = 50,
        n_test_episodes: int = 5,
        n_train_trajectories: int = 4,
        save_best_path: str = "models/policy_network.pt",
        verbose: bool = True,
    ):
        """Main training loop."""
        logs = {"losses": [], "train": [], "test": [], "balanced": []}
        best_avg_PV_test = -np.inf
        prev_avg_PV_test = -np.inf

        for it in range(n_train_iterations):
            # ---- rollouts & loss ----
            total_loss = 0.0
            for _ in range(n_train_trajectories):
                _, _, rewards, _, _ = self.generate_episode(env_name="train")
                total_loss += self.get_policy_loss(rewards, gamma) / n_train_trajectories

            # ---- update ----
            loss_value = self.update_policy(total_loss)

            # ---- evaluate ----
            avg_PV_train, avg_PV_test, avg_PV_bal = self.evaluate_policy(n_test_episodes)

            logs["losses"].append(loss_value)
            logs["train"].append(avg_PV_train)
            logs["test"].append(avg_PV_test)
            logs["balanced"].append(avg_PV_bal)

            if verbose:
                print(
                    f"Iter {it:03d} | Loss: {loss_value:.6f} | "
                    f"PV(train): {avg_PV_train:.4f} | PV(test): {avg_PV_test:.4f} | "
                    f"PV_bal(test): {avg_PV_bal:.4f} | Best PV(test): {best_avg_PV_test:.4f}"
                )

            # Save best policy weights
            if avg_PV_test > prev_avg_PV_test:
                os.makedirs(os.path.dirname(save_best_path) or ".", exist_ok=True)
                self.policy.save(save_best_path)

            # Update best trackers
            best_avg_PV_test = max(best_avg_PV_test, avg_PV_test)
            prev_avg_PV_test = avg_PV_test

            # Save checkpoint (every iter)
            self.save_checkpoint(path="models/checkpoint.pt", it=it, best=best_avg_PV_test)

        return logs


# ---------- tiny smoke test ----------
if __name__ == "__main__":
    agent = REINFORCE(
        n_neurons=[128, 64, 32],
        T=20,
        learning_rate=1e-3,
        data_path="data/datasets",
        lookback=5,
    )

    logs = agent.train_agent(
        gamma=0.99,
        n_train_iterations=2,  # small for quick test
        n_test_episodes=2,
        n_train_trajectories=2,
        save_best_path="models/policy_network.pt",
    )

    print("Done. Last PV(test):", logs["test"][-1])
