import os
import argparse
import pandas as pd
from scripts.reinforce import REINFORCE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=str, default="256,256,256", help="hidden sizes, comma-separated")
    ap.add_argument("--T", type=int, default=90)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--n_test_eps", type=int, default=10)
    ap.add_argument("--n_traj", type=int, default=100)
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--data_path", type=str, default="data/datasets")
    ap.add_argument("--model_path", type=str, default="models/policy_network.pt")
    ap.add_argument("--logs_out", type=str, default="data/processed/reinforce_logs.csv")
    ap.add_argument("--resume", action="store_true", help="resume from checkpoint if available")
    ap.add_argument("--ckpt_path", type=str, default="models/checkpoint.pt", help="path to checkpoint file")
    args = ap.parse_args()

    n_neurons = [int(x) for x in args.layers.split(",")]

    agent = REINFORCE(
        n_neurons=n_neurons,
        T=args.T,
        learning_rate=args.lr,
        data_path=args.data_path,
        lookback=args.lookback,
        device="cpu",
    )

    start_it = 0
    best_test = -float("inf")

    if args.resume and os.path.exists(args.ckpt_path):
        start_it, best_test = agent.load_checkpoint(args.ckpt_path)
        print(f"Resumed from iter {start_it}, best test PV={best_test:.4f}")

    logs = agent.train_agent(
        gamma=args.gamma,
        n_train_iterations=args.iters - start_it,
        n_test_episodes=args.n_test_eps,
        n_train_trajectories=args.n_traj,
        save_best_path=args.model_path,
        verbose=True,
    )




    os.makedirs(os.path.dirname(args.logs_out) or ".", exist_ok=True)
    pd.DataFrame(logs).to_csv(args.logs_out, index=False)
    print(f"Saved model → {args.model_path}")
    print(f"Saved logs  → {args.logs_out}")

if __name__ == "__main__":
    main()
