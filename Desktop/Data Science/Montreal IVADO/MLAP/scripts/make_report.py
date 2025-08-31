# scripts/make_report.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scripts.reinforce import REINFORCE


def pv_grid(PV_df: pd.DataFrame, PVb_df: pd.DataFrame, episodes: int = 10):
    cols = min(5, episodes)
    rows = int(np.ceil(episodes / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    axs = np.array(axs).ravel()

    for i in range(episodes):
        axs[i].plot(PV_df.iloc[:, i], label="Agent")
        axs[i].plot(PVb_df.iloc[:, i], label="Balanced", linestyle="--")
        axs[i].set_title(f"Episode {i+1}")
        if i == 0:
            axs[i].legend(loc="best")

    # remove extra axes if any
    for j in range(episodes, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle("Portfolio Value: Agent vs Balanced", y=0.99)
    fig.tight_layout()
    return fig


def allocations_grid(actions: np.ndarray, asset_names, title="Allocations (weights)"):
    # ensure (T, n_assets) float array
    actions = np.asarray(actions)
    if actions.dtype == object:
        actions = np.vstack(actions)
    actions = actions.astype(float, copy=False)

    actions_df = pd.DataFrame(actions, columns=asset_names)
    n = len(asset_names)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows))
    axs = np.array(axs).ravel()

    for i, c in enumerate(actions_df.columns):
        axs[i].plot(actions_df[c].values)
        axs[i].set_title(c)
        axs[i].set_ylim(0, 1)

    # remove unused axes
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    return fig


def metrics_page(metrics: pd.DataFrame, config: dict):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    tbl_cols = ["episode", "final_PV_agent", "final_PV_bal", "excess"]
    show = metrics[tbl_cols].copy()
    show["episode"] = show["episode"].astype(int) + 1
    show = show.round(4)

    table = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.15)

    cfg_lines = [
        "Report Summary",
        f"Episodes: {config['episodes']}",
        f"T (days): {config['T']}",
        f"Lookback: {config['lookback']}",
        f"Layers: {config['layers']}",
        f"Model: {config['model_path']}",
        f"Data path: {config['data_path']}",
    ]
    ax.set_title("\n".join(cfg_lines), loc="left", y=1.08, fontsize=12)

    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=str, default="256,256,256")
    ap.add_argument("--T", type=int, default=90)
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--model", type=str, default="models/policy_network.pt")
    ap.add_argument("--data_path", type=str, default="data/datasets")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out_pdf", type=str, default="reports/agent_report.pdf")
    ap.add_argument("--metrics_csv", type=str, default="reports/episode_metrics.csv")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    # Load trained policy in predicting mode
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

    # Roll episodes
    PVs, PVbs, all_actions = [], [], []
    for _ in range(args.episodes):
        states, actions, rewards, PV, PV_bal = agent.generate_episode(env_name="test")
        PVs.append(PV)
        PVbs.append(PV_bal)
        all_actions.append(actions)  # list of length T; each element is (n_assets,) array

    PV_df = pd.DataFrame(PVs).T
    PVb_df = pd.DataFrame(PVbs).T

    # Metrics
    final_agent = PV_df.iloc[-1].values
    final_bal = PVb_df.iloc[-1].values
    metrics = pd.DataFrame(
        {
            "episode": np.arange(args.episodes),
            "final_PV_agent": final_agent,
            "final_PV_bal": final_bal,
        }
    )
    metrics["excess"] = metrics["final_PV_agent"] - metrics["final_PV_bal"]
    metrics.to_csv(args.metrics_csv, index=False)

    # Pick episodes for allocation pages (best & median by final PV)
    best_idx = int(np.argmax(final_agent))
    med_idx = int(np.argsort(final_agent)[len(final_agent) // 2])

    config = {
        "episodes": args.episodes,
        "T": args.T,
        "lookback": args.lookback,
        "layers": layers,
        "model_path": args.model,
        "data_path": args.data_path,
    }

    # Create PDF
    with PdfPages(args.out_pdf) as pdf:
        # Page 1: PV grid
        fig1 = pv_grid(PV_df, PVb_df, episodes=args.episodes)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # Page 2: allocations for best episode
        best_actions = np.vstack(all_actions[best_idx])   # (T, n_assets)
        fig2 = allocations_grid(
            best_actions,
            asset_names=agent.env_test.assets_col,
            title=f"Allocations — Best Episode (#{best_idx+1})",
        )
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # Page 3: allocations for median episode
        med_actions = np.vstack(all_actions[med_idx])     # (T, n_assets)
        fig3 = allocations_grid(
            med_actions,
            asset_names=agent.env_test.assets_col,
            title=f"Allocations — Median Episode (#{med_idx+1})",
        )
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        # Page 4: metrics table + config
        fig4 = metrics_page(metrics, config)
        pdf.savefig(fig4, bbox_inches="tight")
        plt.close(fig4)

    print(f"Saved PDF  → {args.out_pdf}")
    print(f"Saved CSV  → {args.metrics_csv}")


if __name__ == "__main__":
    main()
