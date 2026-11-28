"""
A tiny made-up dataset (8 hotels) with the same columns as the real files, so the tests run anywhere
(also in GitHub CI, where the real data is not available).
"""
import os

import pytest

from src.c2.data import HotelData, set_data
from tests.helpers import make_files

os.environ["C2_PRICE_MARGINS"] = "off"      # the tests use the fixed 30% rule (tests/test_price_margins.py: the file)


@pytest.fixture(scope="session")
def fixture_dirs(tmp_path_factory):
    return make_files(tmp_path_factory.mktemp("c2"))


@pytest.fixture(scope="session")
def data(fixture_dirs):
    d = HotelData(*fixture_dirs)
    set_data(d)
    return d


@pytest.fixture(autouse=True)
def dummy_c1(monkeypatch):
    monkeypatch.setenv("C1_MODE", "dummy")