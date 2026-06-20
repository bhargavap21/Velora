"""
Modal cloud runner for StratRL.

Before running, create a Modal secret at modal.com:
  Name: anthropic-secret
  Key:  ANTHROPIC_API_KEY
  Value: your Anthropic API key

Then run with:
  modal run modal_runner.py
"""

from __future__ import annotations

import modal

app = modal.App("stratrl")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "anthropic",
        "pandas",
        "numpy",
        "yfinance",
        "pyarrow",
        "python-dotenv",
        "exa-py",
    )
    .add_local_dir(
        ".", remote_path="/root/stratrl",
        ignore=[".venv", ".venv-modal", "data", "results", "frontend/node_modules", "__pycache__", "*.pyc"],
    )
)


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("anthropic-secret"),
        modal.Secret.from_name("exa-secret"),  # must have EXA_API_KEY set
    ],
    timeout=600,
)
def run_episode_remote(seed_strategy: dict) -> list[dict]:
    """Runs one full episode. Returns a list of event dicts (one per turn, plus a
    final episode_complete dict) — Modal's `.map()` requires a single return value
    per input rather than a generator, so episode_core's generator is buffered here.
    """
    import sys
    sys.path.insert(0, "/root/stratrl")

    from episode_core import run_episode_events

    return list(run_episode_events(seed_strategy))


@app.local_entrypoint()
def main() -> None:
    import json
    from datetime import datetime
    from pathlib import Path

    from episode_core import SEED_STRATEGIES

    print(f"Launching {len(SEED_STRATEGIES)} episodes in parallel...")
    all_events: list[dict] = []
    for episode_events in run_episode_remote.map(SEED_STRATEGIES):
        all_events.extend(episode_events)

    completions = {
        e["seed_name"]: e for e in all_events if e["type"] == "episode_complete"
    }

    _SEP = "━" * 52
    print(f"\nSTRATRL MODAL RUN — {len(completions)} episodes")
    print(_SEP)
    print(f"{'Rank':<5} {'Seed':<22} {'Init Sharpe':>11} {'Final Sharpe':>12} {'Delta':>7} {'Reward':>7} {'Turns':>6}")

    ranked = sorted(
        completions.values(),
        key=lambda r: r["total_reward"],
        reverse=True,
    )

    for rank, r in enumerate(ranked, 1):
        init_sharpe = r["initial_sharpe"]
        final_sharpe = r["final_sharpe"]
        delta = final_sharpe - init_sharpe
        print(
            f"{rank:>4}  {r['seed_name']:<22} {init_sharpe:>11.2f} {final_sharpe:>12.2f} "
            f"{delta:>+7.2f} {r['total_reward']:>7.2f} {r['turns_taken']:>6}"
        )

    print(_SEP)
    best = ranked[0]
    print(f"Best: {best['seed_name']}  {best['initial_sharpe']:.2f} → {best['final_sharpe']:.2f} Sharpe")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"modal_run_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({"events": all_events}, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
