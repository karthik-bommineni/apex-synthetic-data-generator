from datetime import datetime

import numpy as np

from apex.sampler import Sampler
from apex.schema import Column


def _sampler() -> Sampler:
    return Sampler(
        np.random.default_rng(7),
        datetime(2024, 1, 1),
        datetime(2024, 12, 31, 23, 59, 59),
    )


def test_truncated_normal_stays_inside_range() -> None:
    column = Column(
        name="interest_rate",
        type="float",
        strategy="normal",
        min=0.03,
        max=0.18,
        mean=0.09,
        std=0.04,
        precision=4,
    )
    values = _sampler().draw(column, 2000)
    assert len(values) == 2000
    assert min(values) >= 0.03
    assert max(values) <= 0.18


def test_lognormal_money_stays_inside_range() -> None:
    column = Column(
        name="amount",
        type="float",
        strategy="lognormal",
        min=1,
        max=20000,
        mu=4.1,
        sigma=1.05,
        precision=2,
    )
    values = _sampler().draw(column, 1500)
    assert min(values) >= 1
    assert max(values) <= 20000


def test_strictly_increasing_and_decreasing() -> None:
    increasing = Column(
        name="posted_at",
        type="datetime",
        strategy="strictly_increasing",
        scope="within",
        within="account_id",
    )
    groups = ["A", "A", "A", "B", "B"]
    values = _sampler().draw(increasing, len(groups), groups=groups)
    assert values[0] < values[1] < values[2]
    assert values[3] < values[4]

    decreasing = Column(
        name="suspicion_rank",
        type="integer",
        strategy="strictly_decreasing",
        scope="within",
        within="account_id",
        min=1,
        max=50,
    )
    ranks = _sampler().draw(decreasing, len(groups), groups=groups)
    assert ranks[0] > ranks[1] > ranks[2]
    assert ranks[3] > ranks[4]
    assert all(1 <= rank <= 50 for rank in ranks)


def test_correlate_with_tracks_source_direction() -> None:
    source_column = Column(
        name="annual_income",
        type="float",
        strategy="uniform",
        min=20000,
        max=200000,
        precision=2,
    )
    limit = Column(
        name="credit_limit",
        type="float",
        strategy="correlate_with",
        correlate_with="annual_income",
        correlation=0.9,
        min=1000,
        max=75000,
        precision=2,
    )
    sampler = _sampler()
    income = sampler.draw(source_column, 4000)
    limits = sampler.draw(limit, 4000, source=income)
    assert min(limits) >= 1000
    assert max(limits) <= 75000
    correlation = float(np.corrcoef(np.asarray(income, dtype=float), np.asarray(limits, dtype=float))[0, 1])
    assert correlation > 0.6
