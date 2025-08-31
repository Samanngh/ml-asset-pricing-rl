# scripts/evaluate_agent.py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.reinforce import REINFORCE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=str, default="256,256,256")
    ap.add_argument("--T", type=int, default=90)
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--model", type=str, default="models/policy_network.pt")
    ap.add_argument("--data_path", type=str, default="data/datasets")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    agent = REINFORCE(
        n_neurons=layers,
        T=args.T,
        learning_rate=1e-4,
        data_path=args.data_path,
        lookback=args.lookback,
        mode="predicting",
        model_path=args.model,
        device="cpu",
    )

    PVs, PVs_balanced = [], []
    last_actions = None
    for _ in range(args.episodes):
        states, actions, rewards, PV, PV_bal = agent.generate_episode(env_name="test")
        PVs.append(PV)
        PVs_balanced.append(PV_bal)
        last_actions = actions

    PV_df = pd.DataFrame(PVs).T
    PVb_df = pd.DataFrame(PVs_balanced).T

    fig, axs = plt.subplots(2, int(np.ceil(args.episodes/2)), figsize=(16, 6))
    axs = axs.ravel()
    for i in range(args.episodes):
        axs[i].plot(PV_df.iloc[:, i], label="Agent PV")
        axs[i].plot(PVb_df.iloc[:, i], label="Balanced PV", linestyle="--")
        axs[i].set_title(f"Episode {i+1}")
        if i == 0:
            axs[i].legend(loc="best")
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    fig.tight_layout()
    plt.show()

    # allocations for last episode
    col_names = agent.env_test.assets_col
    alloc_df = pd.DataFrame(last_actions, columns=col_names)

    rows = int(np.ceil(len(col_names)/4))
    fig, axs = plt.subplots(rows, 4, figsize=(15, 3*rows))
    axs = axs.ravel()
    for i, c in enumerate(col_names):
        axs[i].plot(alloc_df[c].values)
        axs[i].set_title(c)
        axs[i].set_ylim(0, 1)
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
