import numpy as np
import pandas as pd

from execution_env.simulator.benchmark import (
    _MAX_SLIPPAGE_BPS,
    _UNFILLED_PENALTY_COEF,
    compute_twap,
    compute_vwap,
    execution_vwap,
    slippage_reward,
)
from execution_env.simulator.market_sim import ImpactModel, generate_intraday_path, u_shaped_volume_curve


def test_u_shaped_volume_curve_sums_to_one():
    curve = u_shaped_volume_curve(26)
    assert abs(curve.sum() - 1.0) < 1e-6


def test_u_shaped_volume_curve_is_higher_at_edges():
    curve = u_shaped_volume_curve(26)
    assert curve[0] > curve[len(curve) // 2]
    assert curve[-1] > curve[len(curve) // 2]


def test_u_shaped_volume_curve_edge_to_midday_ratio():
    curve = u_shaped_volume_curve(101)  # odd length so the midpoint lands exactly at x=0
    midday = curve[len(curve) // 2]
    ratio = curve[0] / midday
    assert 2.5 < ratio < 3.5


def test_intraday_path_pinned_to_open_and_close():
    day_row = pd.Series({"Open": 100.0, "High": 105.0, "Low": 98.0, "Close": 102.0})
    rng = np.random.default_rng(0)
    path = generate_intraday_path(day_row, n_slices=26, rng=rng)
    assert path[0] == 100.0
    assert path[-1] == 102.0


def test_intraday_path_stays_within_day_range():
    day_row = pd.Series({"Open": 100.0, "High": 105.0, "Low": 98.0, "Close": 102.0})
    for seed in range(10):
        rng = np.random.default_rng(seed)
        path = generate_intraday_path(day_row, n_slices=26, rng=rng)
        assert path.min() >= 98.0
        assert path.max() <= 105.0


def test_vwap_matches_twap_for_equal_volumes():
    import numpy as np

    prices = np.array([100.0, 101.0, 99.0])
    volumes = np.array([10.0, 10.0, 10.0])
    assert abs(compute_vwap(prices, volumes) - compute_twap(prices)) < 1e-6


def test_slippage_reward_zero_for_exact_match():
    reward = slippage_reward(agent_vwap=100.0, benchmark_vwap=100.0, side="buy", filled_fraction=1.0)
    assert abs(reward) < 1e-6


def test_slippage_reward_penalizes_unfilled_inventory():
    full = slippage_reward(agent_vwap=100.0, benchmark_vwap=100.0, side="buy", filled_fraction=1.0)
    partial = slippage_reward(agent_vwap=100.0, benchmark_vwap=100.0, side="buy", filled_fraction=0.5)
    assert partial < full


def test_unfilled_order_worse_than_worst_case_slippage():
    # Zero fill, but with the best possible (zero) slippage.
    unfilled_reward = slippage_reward(agent_vwap=100.0, benchmark_vwap=100.0, side="buy", filled_fraction=0.0)
    # Full fill, but with the worst-case (clipped) slippage.
    worst_case_agent_vwap = 100.0 * (1 + _MAX_SLIPPAGE_BPS / 10_000)
    worst_slippage_reward = slippage_reward(
        agent_vwap=worst_case_agent_vwap, benchmark_vwap=100.0, side="buy", filled_fraction=1.0
    )
    assert unfilled_reward < worst_slippage_reward


def test_slippage_is_clipped_to_max_bps():
    reward_at_clip = slippage_reward(
        agent_vwap=100.0 * (1 + _MAX_SLIPPAGE_BPS / 10_000), benchmark_vwap=100.0, side="buy", filled_fraction=1.0
    )
    reward_beyond_clip = slippage_reward(agent_vwap=200.0, benchmark_vwap=100.0, side="buy", filled_fraction=1.0)
    assert abs(reward_at_clip - reward_beyond_clip) < 1e-9


def test_reward_is_bounded():
    max_possible_reward = (_MAX_SLIPPAGE_BPS + _UNFILLED_PENALTY_COEF) / 100.0
    for agent_vwap in [50.0, 100.0, 150.0, 1000.0]:
        for filled_fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for side in ["buy", "sell"]:
                reward = slippage_reward(
                    agent_vwap=agent_vwap, benchmark_vwap=100.0, side=side, filled_fraction=filled_fraction
                )
                assert abs(reward) <= max_possible_reward + 1e-9


def test_impact_model_grows_with_participation():
    model = ImpactModel(adv=100_000)
    small = model.temporary_impact(1_000)
    large = model.temporary_impact(50_000)
    assert large > small


def test_impact_model_is_small_for_reasonable_order_size():
    model = ImpactModel(adv=100_000)
    # Impact is driven by the *per-slice* participation rate (qty / slice_volume), so a
    # "reasonable order" is one that consumes a small fraction of the slice it trades into.
    # A volume-aware schedule keeps participation low (~1% of the slice) -> a few bps.
    slice_volume = 100_000 / 26  # a representative slice's volume
    qty = 0.01 * slice_volume  # 1% participation
    temp = model.temporary_impact(qty, slice_volume)
    perm = model.permanent_impact(qty, slice_volume)
    assert temp < 0.0005  # < 5bps
    assert perm < 0.0005  # < 5bps


def test_impact_model_grows_superlinearly_for_outsized_orders():
    model = ImpactModel(adv=100_000)
    # Linear scaling from 5% to 50% ADV would give exactly 10x. The convexity multiplier
    # should push outsized orders well past that.
    temp_5pct = model.temporary_impact(5_000)
    temp_50pct = model.temporary_impact(50_000)
    perm_5pct = model.permanent_impact(5_000)
    perm_50pct = model.permanent_impact(50_000)
    assert temp_50pct / temp_5pct > 10
    assert perm_50pct / perm_5pct > 10
