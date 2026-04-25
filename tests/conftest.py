"""Pytest configuration for the ``tests/`` tree."""

def pytest_configure(config):
    from rl import _win_triton_warnings

    _win_triton_warnings.apply()
    config.addinivalue_line("markers", "slow: end-to-end or heavy subprocess tests")
