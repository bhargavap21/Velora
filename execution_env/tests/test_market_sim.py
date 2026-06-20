from execution_env.simulator.benchmark import compute_twap, compute_vwap, execution_vwap, slippage_reward
from execution_env.simulator.market_sim import ImpactModel, u_shaped_volume_curve


def test_u_shaped_volume_curve_sums_to_one():
    curve = u_shaped_volume_curve(26)
    assert abs(curve.sum() - 1.0) < 1e-6


def test_u_shaped_volume_curve_is_higher_at_edges():
    curve = u_shaped_volume_curve(26)
    assert curve[0] > curve[len(curve) // 2]
    assert curve[-1] > curve[len(curve) // 2]


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


def test_impact_model_grows_with_participation():
    model = ImpactModel(adv=100_000)
    small = model.temporary_impact(1_000)
    large = model.temporary_impact(50_000)
    assert large > small
