Project Structure


MLAP/
├── data/
│   ├── raw/          # Cached downloads from yfinance
│   ├── processed/    # Train/test splits, prepared CSV/Parquet
│   └── datasets/     # train.csv / test.csv for RL env
├── models/           # Saved policy checkpoints
├── scripts/
│   ├── env.py        # Environment (portfolio simulator)
│   ├── policy.py     # Policy network definitiona
│   ├── reinforce.py  # REINFORCE agent
│   ├── fetch_daily_returns_yf.py  # Data download + preprocessing
│   ├── train_policy.py  # Training loop entrypoint
│   └── evaluate_agent.py # Evaluation + plots
└── README.md




Getting Started

Clone repo

git clone https://github.com/Samanngh/ml-asset-pricing-rl.git
cd ml-asset-pricing-rl


Install dependencies

pip install -r requirements.txt


Build dataset

python -m scripts.fetch_daily_returns_yf


Downloads daily returns for 16 global assets and prepares train/test splits.

Train agent

python -m scripts.train_policy \
  --layers 256,256,256 \
  --T 90 \
  --lr 0.0001 \
  --gamma 0.99 \
  --iters 2000 \
  --n_test_eps 10 \
  --n_traj 100


Evaluate agent

python -m scripts.evaluate_agent --T 90 --episodes 10


Generates performance plots of portfolio vs balanced benchmark, and allocation dynamics.




Results 

The agent learns to rebalance portfolio allocations dynamically.

Outperforms balanced benchmark on some test episodes.

Visualizations of PV curves and allocations are included.



Tech Stack

Python 3.12+

PyTorch (policy + training)

Pandas / Numpy (data handling)

Matplotlib (visualization)

yfinance (market data)
