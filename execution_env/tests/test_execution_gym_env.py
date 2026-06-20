import numpy as np
from gymnasium.utils.env_checker import check_env

from execution_env.rl.execution_gym_env import ExecutionEnv


def test_check_env_passes():
    check_env(ExecutionEnv(), skip_render_check=True)


def test_constant_full_participation_completes_order():
    env = ExecutionEnv()
    obs, _ = env.reset(seed=0)
    for _ in range(env._n_slices):
        obs, reward, terminated, truncated, _ = env.step(np.array([1.0], dtype=np.float32))
        if terminated or truncated:
            break
    assert abs(env._shares_remaining) < 1e-6


def test_action_never_oversells():
    env = ExecutionEnv()
    obs, _ = env.reset(seed=1)
    for _ in range(env._n_slices):
        obs, reward, terminated, truncated, _ = env.step(np.array([2.0], dtype=np.float32))
        assert env._shares_remaining >= 0.0
        if terminated or truncated:
            break
    assert abs(env._shares_remaining) < 1e-6


def test_observation_shape_and_bounds():
    env = ExecutionEnv()
    obs, _ = env.reset(seed=2)
    assert obs.shape == (5,)
    assert env.observation_space.contains(obs)
    for _ in range(env._n_slices):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        assert obs.shape == (5,)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break
