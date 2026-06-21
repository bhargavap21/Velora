"""
Run PPO training on Modal instead of the local machine: primes the data cache for the
expanded TRAIN_TICKERS universe (once, persisted on a Volume) and trains a longer run
with more CPU than a laptop comfortably offers, since the per-episode minute-bar
processing that dominates rollout wall-clock parallelizes across cores.

One-time setup (see execution_env/README.md for the full walkthrough):
    pip install modal
    modal setup                                                  # browser auth
    modal secret create alpaca-creds \\
        ALPACA_API_KEY=... ALPACA_SECRET_KEY=...

Usage:
    modal run execution_env/rl/modal_train.py

On completion, writes the checkpoint + training-curve PNG to the *local*
execution_env/results/ directory -- server.py's _get_ppo_model() picks up the new
checkpoint automatically, no other code change needed.
"""

from __future__ import annotations

import modal

app = modal.App("velora-ppo-train")

_REPO_ROOT = "/root/velora"
# Separate from execution_env/data_cache/ -- that path is baked into the image by
# add_local_dir() below (it's the committed parquet cache), and Modal refuses to mount a
# Volume on a non-empty path. market_sim._DATA_DIR is redirected here at the top of
# train() instead, so priming/training read and write Volume-persisted data, not the
# image's baked-in copy.
_DATA_VOLUME_PATH = "/cache/data_cache"

# Deliberately not the full requirements.txt: that file pins protobuf==7.35.1 for
# unrelated tools (anthropic, hud-python, ...) which conflicts with modal's own
# protobuf<7.0 requirement. Training only needs this handful of packages -- a smaller,
# conflict-free, faster-to-build image.
_TRAIN_DEPENDENCIES = [
    "numpy==2.4.6",
    "pandas==3.0.3",
    "pyarrow",
    "gymnasium",
    "stable-baselines3",
    "matplotlib==3.11.0",
    "alpaca-py==0.43.4",
    "yfinance==1.4.1",
    "python-dotenv==1.2.2",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Install CPU-only torch *before* stable-baselines3 -- otherwise pip resolves a full
    # CUDA build (hundreds of MB of nvidia-* wheels) for a function that only requests
    # CPU (cpu=16 below, no GPU). Installing it first means stable-baselines3 just finds
    # torch already satisfied and skips pulling the GPU variant.
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(*_TRAIN_DEPENDENCIES)
    .add_local_dir(".", remote_path=_REPO_ROOT)
)

data_volume = modal.Volume.from_name("velora-data-cache", create_if_missing=True)

# Bigger run than the local default (more diverse data needs more samples to converge)
# and more parallel envs than a laptop's core count comfortably sustains.
_MODAL_N_ENVS = 16
_MODAL_TOTAL_TIMESTEPS = 2_000_000


@app.function(
    image=image,
    volumes={_DATA_VOLUME_PATH: data_volume},
    secrets=[modal.Secret.from_name("alpaca-creds")],
    cpu=16,
    timeout=4 * 60 * 60,
)
def train() -> dict[str, bytes]:
    import sys
    from pathlib import Path

    sys.path.insert(0, _REPO_ROOT)

    import execution_env.simulator.market_sim as market_sim

    # Redirect the data cache to the Volume-mounted path (see _DATA_VOLUME_PATH above)
    # instead of the image's baked-in execution_env/data_cache/ copy, so priming here
    # persists across runs via the Volume.
    market_sim._DATA_DIR = Path(_DATA_VOLUME_PATH)
    market_sim._DATA_DIR.mkdir(parents=True, exist_ok=True)

    from execution_env.rl.prime_data_cache import prime
    from execution_env.rl.train_ppo import (
        MODEL_PATH,
        CURVE_PATH,
        TRAIN_TICKERS,
        _HOLDOUT_DAY_RANGE,
        _N_EVAL_EPISODES,
        build_envs,
        print_holdout_report,
        run_holdout_eval,
        run_training,
        save_checkpoint_and_curve,
    )
    from stable_baselines3 import PPO

    print(f"Priming data cache for {len(TRAIN_TICKERS)} tickers...")
    succeeded, failed = prime(TRAIN_TICKERS)
    data_volume.commit()  # persist the primed cache before training reads it
    if failed:
        print(f"Skipped (no data from any provider): {failed}")
    tickers = succeeded

    train_env = build_envs(tickers=tickers, n_envs=_MODAL_N_ENVS)
    model = PPO("MlpPolicy", train_env, verbose=1)

    print(f"Training PPO for {_MODAL_TOTAL_TIMESTEPS} timesteps across {_MODAL_N_ENVS} envs "
          f"on {len(tickers)} tickers...")
    rewards = run_training(model, _MODAL_TOTAL_TIMESTEPS)
    print(f"Completed {len(rewards)} training episodes (env 0).")

    save_checkpoint_and_curve(model, rewards)

    eval_seeds = list(range(10_000, 10_000 + _N_EVAL_EPISODES))
    ppo_rewards, vwap_rewards, naive_rewards = run_holdout_eval(model, tickers=tickers, eval_seeds=eval_seeds)
    print_holdout_report(ppo_rewards, vwap_rewards, naive_rewards, _HOLDOUT_DAY_RANGE, (0.02, 0.10))

    return {
        "checkpoint": MODEL_PATH.read_bytes(),
        "curve": CURVE_PATH.read_bytes(),
    }


@app.local_entrypoint()
def main() -> None:
    from pathlib import Path

    artifacts = train.remote()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "ppo_execution.zip").write_bytes(artifacts["checkpoint"])
    (results_dir / "execution_ppo_curve.png").write_bytes(artifacts["curve"])
    print(f"Wrote checkpoint + curve to {results_dir}")
