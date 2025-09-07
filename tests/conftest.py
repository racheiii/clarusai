import os, random
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def _seed_everything(monkeypatch):
    random.seed(42)
    np.random.seed(42)
    monkeypatch.setenv("PYTHONHASHSEED", "42")